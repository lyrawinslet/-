"""Reopen saved results; independently check arithmetic and information boundaries."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_experiment as run
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'full_run')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args(); out=args.output
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    project=run.DEFAULT_PROJECT
    code=project/'code/q3_model.py'
    spec=importlib.util.spec_from_file_location('q3_verifier_source',code)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    pack=project/'comparison'
    old=pd.read_csv(pack/'q3_main_interval_ledger.csv.gz')
    checks={}
    frames={}
    for arm in ['greedy_replay','price_aware_mpc']:
        f=pd.read_csv(out/f'{arm}_interval.csv.gz');frames[arm]=f
        assert len(f)==334*144
        assert not f[['date','interval_index']].duplicated().any()
        np.testing.assert_array_equal(f[['date','interval_index']],old[['date','interval_index']])
        flow=f.contract_kwh+f.emergency_kwh+f.discharge_kwh-f.actual_net_kwh-f.charge_kwh-f.unused_kwh
        soc=f.soc_start_kwh+.9*f.charge_kwh-f.discharge_kwh/.9
        gap=f.actual_net_kwh-f.contract_kwh
        limits=max(0.,1200-f.soc_end_kwh.min(),f.soc_end_kwh.max()-10800,
                   f.charge_kwh.max()-5000/6,f.discharge_kwh.max()-5000/6,
                   float((f.discharge_kwh-np.maximum(gap,0)).max()))
        report={'balance_max_kwh':float(abs(flow).max()),
          'state_max_kwh':float(abs(soc-f.soc_end_kwh).max()),
          'state_chain_max_kwh':float(abs(f.soc_start_kwh.to_numpy()[1:]-f.soc_end_kwh.to_numpy()[:-1]).max()),
          'constraint_violation_kwh':float(limits),
          'overlap_kwh':float(np.minimum(f.charge_kwh,f.discharge_kwh).max()),
          'emergency_cost_error_yuan':float(abs(5*f.price_yuan_per_kwh*f.emergency_kwh-f.emergency_cost_yuan).max()),
          'fixed_cost_error_yuan':float(abs(f.contract_cost_yuan-old.contract_total_cost_yuan).max()),
          'delivered_contract_error_kwh':float(abs(f.contract_kwh-old.final_contract_kwh).max()),
          'greedy_charge_error_kwh':float(abs(f.charge_kwh-np.minimum(np.minimum(np.maximum(-gap,0),5000/6),np.maximum((10800-f.soc_start_kwh)/.9,0))).max())}
        record=next(x for x in summary['arms'] if x['arm']==arm)
        for field in ['actual_total_cost_yuan','emergency_cost_yuan','emergency_kwh','unused_kwh']:
            report[f'{field}_summary_error']=abs(float(f[field].sum())-record[field])
        assert max(report.values())<5e-5,report
        assert (f[['charge_kwh','discharge_kwh','emergency_kwh','unused_kwh']].min()>=-1e-8).all()
        checks[arm]=report
    assert frames['greedy_replay'].soc_start_kwh.iloc[0]==frames['price_aware_mpc'].soc_start_kwh.iloc[0]
    saved_diff=float(frames['greedy_replay'].actual_total_cost_yuan.sum()-frames['price_aware_mpc'].actual_total_cost_yuan.sum())
    assert abs(saved_diff-summary['saving_yuan'])<5e-5
    print('Saved ledgers and summary independently checked.',flush=True)

    # Rebuild three causal decision inputs after strongly perturbing information
    # unavailable at those instants. Past errors and current observation are held.
    data=m.load_data(); archive=m.build_forecast_archive(data,'official')
    events=pd.read_csv(pack/'q3_adjustment_events.csv.gz')
    causal=[]
    for day,t in [('2025-03-20',65),('2025-06-21',65),('2025-09-23',65)]:
        d=[x.isoformat() for x in data.dates].index(day);k=t//36
        load=data.load_kw.copy();pv=data.pv_kw.copy();official=data.official_hourly_kw.copy()
        load[d,t+1:]+=10000.;load[d+1:]+=10000.
        pv[d,t+1:]*=1.7;pv[d+1:]*=1.7
        official[d,k+1:]+=9999.;official[d+1:]+=9999.
        altered=replace(data,load_kw=load,pv_kw=pv,net_load_kwh=(load-pv)/6,official_hourly_kw=official)
        altarchive=m.build_forecast_archive(altered,'official')
        prior=np.arange(max(1,d-28),d)
        def margin(a):
            return np.maximum(0,np.quantile(a.errors_kwh[prior,k,t:],.8,axis=0,method='linear'))
        def known_contract(ev):
            q=old[old.date==day].original_contract_kwh.to_numpy().copy()
            for hour in [6,12,18]:
                if hour>k*6: break
                rev=ev[(ev.date==day)&(ev.release_hour==hour)]
                q[rev.interval_index.to_numpy(int)-1]=rev.new_contract_kwh
            return q
        modified_events=events.copy()
        modified_events.loc[(events.date>day)|((events.date==day)&(events.release_hour>k*6)),'new_contract_kwh']+=1234.
        q=known_contract(events);qa=known_contract(modified_events)
        baseline_input=archive.net_kwh[d,k,t:]+margin(archive)-q[t:]
        altered_input=altarchive.net_kwh[d,k,t:]+margin(altarchive)-qa[t:]
        baseline_input[0]=data.net_load_kwh[d,t]-q[t]
        altered_input[0]=altered.net_load_kwh[d,t]-qa[t]
        maximum=float(np.max(abs(baseline_input-altered_input)))
        assert maximum<1e-10, {'date':day,'max_difference':maximum}
        assert np.max(abs(altered.net_load_kwh[d,t+1:]-data.net_load_kwh[d,t+1:]))>100.
        assert np.nanmax(abs(archive.net_kwh[d,k+1]-altarchive.net_kwh[d,k+1]))>100.
        v=.5*.9*np.median(data.price)
        action,_=run.choose_discharge(baseline_input,data.price[t:],6000.,v,True)
        changed,_=run.choose_discharge(altered_input,data.price[t:],6000.,v,True)
        assert abs(action-changed)<1e-10
        causal.append({'date':day,'interval_index':t+1,'available_release_hour':k*6,
                       'future_information_perturbed':True,'controller_input_max_difference':maximum,
                       'action_difference_kwh':abs(action-changed)})
    hashes=protocol.get('packaged_input_hashes',protocol.get('input_hashes',{}))
    if not hashes:raise RuntimeError('Input hash record is missing')
    unchanged=all(m.sha256(project/p)==h for p,h in hashes.items())
    assert unchanged
    run.write_json(args.report or out/'independent_verification.json',{'status':'PASS','arithmetic_checks':checks,
                   'causality_checks':causal,'original_inputs_unchanged':True,
                   'verifier_sha256':m.sha256(Path(__file__)),
                   'boundary':'Checks fixed-contract execution; does not certify a different endogenous procurement policy.'})
    print('PASS arithmetic, physical constraints, and future-information perturbation checks.',flush=True)


if __name__=='__main__':main()
