"""Independent re-read of saved forecast evidence and policy ledgers."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'code'))
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'results')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    out=args.output;s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    a=np.load(out/'forecast_evidence.npz')
    fc=pd.read_csv(out/'forecast_comparison.csv');coverage=pd.read_csv(out/'coverage.csv')
    gamma=pd.read_csv(out/'gamma_comparison.csv');checks={}
    for mode in ['adopted','seasonal_persistence','equal_window_mean']:
        row=fc[fc['mode']==mode].iloc[0]
        le=a['q2_actual_load']-a[mode+'_load'];pe=a['q2_actual_pv']-a[mode+'_pv']
        net=(a[mode+'_load']-a[mode+'_pv'])/6;e=a['q2_actual_net']-net;b=a[mode+'_margin']
        expected=[abs(le).mean(),abs(pe).mean(),abs(e).mean(),np.sqrt((e**2).mean()),e.mean(),100*(e<=b+1e-9).mean()]
        actual=row[['load_mae_kw','pv_mae_kw','net_mae_kwh','net_rmse_kwh','net_bias_kwh','coverage_percent']].to_numpy(float)
        checks['forecast_'+mode]=float(np.max(abs(actual-expected)))
        assert checks['forecast_'+mode]<1e-8
    e2=a['q2_actual_net']-(a['adopted_load']-a['adopted_pv'])/6;b2=a['adopted_margin']
    e3=a['q3_error_latest'];b3=a['q3_margin_latest']
    assert np.max(abs(a['q2_actual_net']-a['q3_forecast_latest']-e3))<1e-9
    months=np.array([str(x)[:7] for x in pd.date_range('2025-02-01','2025-12-31')])
    for _,row in coverage.iterrows():
        question=row.question;label=row['group'];scope=row.scope
        error,margin=(e2,b2) if question=='Q2' else (e3,b3)
        if scope=='month':error,margin=error[months==label],margin[months==label]
        elif scope=='4h_group':
            start=int(label[:2])*6;error,margin=error[:,start:start+24],margin[:,start:start+24]
        elif scope in ['all_remaining','executed_6h']:
            hour=int(label.split('_')[1]);k=hour//6
            section=slice(hour*6,144 if scope=='all_remaining' else hour*6+36)
            error=a['q3_all_errors'][:,k,section];margin=a['q3_all_margins'][:,k,section]
        assert error.size==row.n and np.isfinite(error).all()
        covered=int((error<=margin+1e-9).sum());assert covered==row.covered_n
        checks['coverage_'+question+'_'+label]=abs(100*covered/error.size-row.coverage_percent)
        assert checks['coverage_'+question+'_'+label]<1e-8
    ledger_checks={}
    for path in sorted(out.glob('*_interval.csv.gz')):
        f=pd.read_csv(path);assert len(f)==334*144
        assert not f[['date','interval_index']].duplicated().any()
        q3=path.name.startswith('q3_')
        q=f.final_contract_kwh if q3 else f.planned_purchase_kwh
        e=f.actual_emergency_kwh if q3 else f.emergency_purchase_kwh
        ch=f.actual_charge_kwh;dis=f.actual_discharge_kwh;unused=f.actual_unused_kwh
        price=f.price_yuan_per_kwh;net=f.actual_net_load_kwh
        contract=f.contract_total_cost_yuan if q3 else f.contract_cost_yuan
        fees=contract+5*price*e
        errors={'balance':float(abs(q+e+dis-net-ch-unused).max()),
          'soc':float(abs(f.soc_end_kwh-f.soc_start_kwh-.9*ch+dis/.9).max()),
          'chain':float(abs(f.soc_start_kwh.to_numpy()[1:]-f.soc_end_kwh.to_numpy()[:-1]).max()),
          'bounds':float(max(0,1200-f.soc_end_kwh.min(),f.soc_end_kwh.max()-10800,ch.max()-5000/6,dis.max()-5000/6)),
          'overlap':float(np.minimum(ch,dis).max()),
          'emergency_fee':float(abs(f.emergency_cost_yuan-5*price*e).max())}
        if not q3:errors['normal_fee']=float(abs(contract-price*q).max())
        else:errors['total_fee']=float(abs(f.actual_total_cost_yuan-fees).max())
        assert max(errors.values())<5e-5,(path,errors)
        if 'gamma' in path.name:
            g=float(path.name.split('gamma_')[1].split('_interval')[0]);question='Q3' if q3 else 'Q2'
            row=gamma[(gamma.question==question)&(gamma.gamma==g)].iloc[0]
        else:
            mode=path.name.removeprefix('q2_').removesuffix('_interval.csv.gz')
            row=fc[fc['mode']==mode].iloc[0]
        errors['aggregate_fee']=abs(float(fees.sum())-row.actual_total_cost_yuan)
        errors['aggregate_emergency']=abs(float(e.sum())-row.emergency_kwh)
        assert max(errors.values())<5e-5,(path,errors)
        ledger_checks[path.name]=errors
    verified={'status':'PASS','forecast_and_coverage_checks':checks,'ledger_arithmetic':ledger_checks,
       'causality_source':'summary.json','boundary':'Causal historical replay; no claim of untouched test-set validation or global parameter optimality.'}
    (args.report or out/'independent_verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS independent forecast metrics, coverage counts, and saved ledger arithmetic.',flush=True)


if __name__=='__main__':main()
