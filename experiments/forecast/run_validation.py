"""Causal forecast baselines, empirical margin coverage, and inherited gamma checks."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'code'))
import numpy as np
import pandas as pd
import q2_common as c
from q2_forecast_experiment import forecast, audit
from q2_full_feedback import run_full_feedback_strategy, MAIN_CONFIG


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def save_csv(path,frame):
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10f')


def baseline_forecast(data, mode, main):
    if mode=='adopted':return main
    load=np.full_like(data.load_kw,np.nan);pv=load.copy()
    for d in range(1,365):
        if mode=='seasonal_persistence':
            load[d]=data.load_kw[d-7 if d>=7 else d-1]
            pv[d]=data.pv_kw[d-1]
        elif mode=='equal_window_mean':
            same=[i for i in range(max(0,d-28),d) if data.dates[i].weekday()==data.dates[d].weekday()]
            if not same:
                same=[i for i in range(max(0,d-28),d) if (data.dates[i].weekday()>=5)==(data.dates[d].weekday()>=5)]
            if not same:same=list(range(max(0,d-7),d))
            load[d]=data.load_kw[same].mean(axis=0)
            pv[d]=data.pv_kw[max(0,d-7):d].mean(axis=0)
        else:raise ValueError(mode)
    net=(load-pv)/6
    return replace(main,load_kw=load,pv_kw=pv,net_load_kwh=net,
                   errors_kwh=data.net_load_kwh-net,weight_mode=mode)


def forecast_metrics(data,f,margin):
    le=data.load_kw[31:]-f.load_kw[31:];pe=data.pv_kw[31:]-f.pv_kw[31:]
    e=f.errors_kwh[31:];b=margin[31:]
    return dict(n=int(e.size),load_mae_kw=float(abs(le).mean()),pv_mae_kw=float(abs(pe).mean()),
      net_mae_kwh=float(abs(e).mean()),net_rmse_kwh=float(np.sqrt((e**2).mean())),
      net_bias_kwh=float(e.mean()),net_error_q80_kwh=float(np.quantile(e,.8)),
      coverage_percent=float(100*np.mean(e<=b+1e-9)),
      excess_mean_kwh=float(np.maximum(e-b,0).mean()))


def coverage_row(question,label,e,b,**extra):
    e=np.asarray(e).ravel();b=np.asarray(b).ravel()
    covered=e<=b+1e-9
    return dict(question=question,group=label,n=int(len(e)),covered_n=int(covered.sum()),
       coverage_percent=float(100*covered.mean()),exceedance_percent=float(100*(~covered).mean()),
       mae_kwh=float(abs(e).mean()),mean_margin_kwh=float(b.mean()),
       excess_mean_kwh=float(np.maximum(e-b,0).mean()),**extra)


def independent_q2_margin(f):
    b=np.zeros((365,144))
    for d in range(8,365):
        ix=np.arange(max(1,d-28),d);ix=ix[f.valid[ix]]
        if len(ix)<7:continue
        for g in range(6):
            # Scalar independent call on explicitly enumerated historic errors.
            values=[f.errors_kwh[i,j] for i in ix for j in range(24*g,24*g+24)]
            b[d,g*24:g*24+24]=max(0.,float(np.quantile(values,.8)))
    return b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    p2=PROJECT/'comparison'
    p3=PROJECT/'comparison'
    q3_path=PROJECT/'code/q3_model.py'
    spec=importlib.util.spec_from_file_location('q3_parameter_validation',q3_path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    paths=[Path(__file__),Path(c.__file__),PROJECT/'code/q2_full_feedback.py',
      PROJECT/'code/q2_forecast_experiment.py',q3_path,c.PRICE_PATH,c.ACTUAL_PATH,m.FORECAST_PATH,
      p2/'q2_main_interval_ledger.csv.gz',p3/'q3_main_interval_ledger.csv.gz',
      p3/'sensitivity_comparison.csv']
    hashes={p.resolve().relative_to(PROJECT).as_posix():c.sha256(p) for p in paths}
    protocol=dict(status='DIAGNOSTIC_ONLY_NO_OFFICIAL_REPLACEMENT',evaluation=['2025-02-01','2025-12-31'],
      forecasting='one-day-at-a-time causal replay; every target uses only earlier days, Q3 uses released forecasts',
      baselines={'seasonal_persistence':'load previous week same slot; PV previous day same slot; before day7 load yesterday',
                 'equal_window_mean':'same weekdays within preceding28d load mean, preceding7d PV mean; no exponential weights'},
      adopted='same weekdays preceding28d load and preceding7d PV, exp(-age/7) weights',
      margin='alpha=.8; each predictor reconstructs prior forecast errors using unchanged Q2 four-hour pooling',
      gamma_values=[0.,.5,1.],gamma_default=.5,
      initialization='all policy runs start Jan1 at6000kWh and carry own realized SOC; costs evaluated Feb-Dec',
      gamma_scope='Q2 and Q3 full policy reruns; forecasts and margins fixed within each question',
      coverage='fraction of actual net-load energy <= issued point forecast + nonnegative historical margin; not supply reliability',
      q3_coverage_scope='both latest forecast actually used in each6h block, and each release on all remaining same-day slots',
      no_new_parameter_selection=True,
      limitations=['Historical dataset has already informed method choices; this is causal retrospective validation, not a pristine holdout.',
                   'Jan1 state is common; different Feb1 states are consequences of January policy operation and must be reported.'],
      source_hashes=hashes)
    save_json(out/'protocol.json',protocol)
    print('Loading Q2 and computing predeclared forecast baselines...',flush=True)
    data=c.load_q2_data();main_f=forecast(data,'weekday')
    modes=['adopted','seasonal_persistence','equal_window_mean']
    fs={mode:baseline_forecast(data,mode,main_f) for mode in modes}
    bs={mode:c.build_risk_margins(fs[mode],.8) for mode in modes}
    old2=pd.read_csv(p2/'q2_main_interval_ledger.csv.gz')
    reg={}
    for col,arr in [('forecast_load_kw',main_f.load_kw),('forecast_pv_kw',main_f.pv_kw),
                     ('point_forecast_net_kwh',main_f.net_load_kwh),('risk_margin_kwh',bs['adopted'])]:
        reg[col]=float(np.nanmax(abs(old2[col].to_numpy()-arr.ravel())))
    assert max(reg.values())<2e-5,reg
    reg['independent_margin']=float(abs(independent_q2_margin(main_f)-bs['adopted']).max())
    assert reg['independent_margin']<1e-9
    forecast_rows=[];runs={};validations={};gamma_rows=[]
    for mode in modes:
        print('Q2 forecast policy '+mode,flush=True)
        result=run_full_feedback_strategy(data,fs[mode],bs[mode],replace(MAIN_CONFIG,strategy_id='forecast_'+mode))
        checks=audit(result.ledger,result.daily,data)
        assert checks['all_pass'] and result.validation['all_pass']
        validations['Q2_'+mode]={'engine':result.validation,'independent':checks}
        runs[mode]=result
        row=forecast_metrics(data,fs[mode],bs[mode]);row.update(mode=mode,
           actual_total_cost_yuan=result.metrics['actual_total_cost_yuan'],
           emergency_kwh=result.metrics['emergency_purchase_kwh'],
           initial_soc_feb1_kwh=float(result.daily.loc[result.daily.date=='2025-02-01','initial_soc_kwh'].iloc[0]))
        forecast_rows.append(row)
        save_csv(out/f'q2_{mode}_daily.csv',result.daily)
        ev=result.ledger[result.ledger.date>='2025-02-01']
        ev.to_csv(out/f'q2_{mode}_interval.csv.gz',index=False,encoding='utf-8-sig',float_format='%.10f',compression='gzip')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    # Only gamma changes here; the selected forecast and its margins are reused.
    for gamma in [0.,.5,1.]:
        print(f'Q2 gamma {gamma}',flush=True)
        r=runs['adopted'] if gamma==.5 else run_full_feedback_strategy(data,main_f,bs['adopted'],replace(MAIN_CONFIG,gamma=gamma,strategy_id=f'gamma_{gamma}'))
        check=audit(r.ledger,r.daily,data);assert check['all_pass'] and r.validation['all_pass']
        validations[f'Q2_gamma_{gamma}']={'engine':r.validation,'independent':check}
        ev=r.ledger[r.ledger.date>='2025-02-01'];daily=r.daily[r.daily.date>='2025-02-01']
        gamma_rows.append(dict(question='Q2',gamma=gamma,actual_total_cost_yuan=r.metrics['actual_total_cost_yuan'],
          emergency_kwh=r.metrics['emergency_purchase_kwh'],initial_soc_feb1_kwh=float(ev.soc_start_kwh.iloc[0]),
          terminal_soc_kwh=float(ev.soc_end_kwh.iloc[-1]),mean_carryover_soc_kwh=float(daily.terminal_soc_kwh.iloc[:-1].mean()),
          upper_carryover_days=int((daily.terminal_soc_kwh.iloc[:-1]>=10800-1e-4).sum())))
        save_csv(out/f'q2_gamma_{gamma}_daily.csv',r.daily)
        if gamma!=.5:ev.to_csv(out/f'q2_gamma_{gamma}_interval.csv.gz',index=False,encoding='utf-8-sig',float_format='%.10f',compression='gzip')
        print(json.dumps(gamma_rows[-1]),flush=True)
    cols=['planned_purchase_kwh','actual_charge_kwh','actual_discharge_kwh','emergency_purchase_kwh','soc_end_kwh','contract_cost_yuan','emergency_cost_yuan']
    reg['q2_baseline_ledger']=float(np.max(abs(runs['adopted'].ledger[cols].to_numpy()-old2[cols].to_numpy())))
    assert reg['q2_baseline_ledger']<2e-5
    save_csv(out/'forecast_comparison.csv',pd.DataFrame(forecast_rows))

    print('Rebuilding Q3 released forecasts and risk archive...',flush=True)
    d3=m.load_data();a3=m.build_forecast_archive(d3,'official');b3=m.build_risk_margins(a3,.8)
    old3=pd.read_csv(p3/'q3_main_interval_ledger.csv.gz')
    er3=np.empty((334,144));mr3=er3.copy();fr3=er3.copy()
    cov=[]
    for k,hour in enumerate([0,6,12,18]):
        sl=slice(k*36,(k+1)*36)
        er3[:,sl]=a3.errors_kwh[31:,k,sl];mr3[:,sl]=b3[31:,k,sl];fr3[:,sl]=a3.net_kwh[31:,k,sl]
        cov.append(coverage_row('Q3',f'release_{hour}_remaining',a3.errors_kwh[31:,k,hour*6:],b3[31:,k,hour*6:],scope='all_remaining'))
        cov.append(coverage_row('Q3',f'release_{hour}_executed',a3.errors_kwh[31:,k,sl],b3[31:,k,sl],scope='executed_6h'))
    reg['q3_forecast']=float(np.max(abs(fr3.ravel()-old3.forecast_net_load_kwh)))
    reg['q3_margin']=float(np.max(abs(mr3.ravel()-old3.risk_margin_kwh)))
    assert max(reg.values())<2e-5
    e2=main_f.errors_kwh[31:];b2=bs['adopted'][31:]
    cov.append(coverage_row('Q2','overall',e2,b2,scope='unique_intervals'))
    cov.append(coverage_row('Q3','overall',er3,mr3,scope='unique_intervals'))
    for g in range(6):
        cov.append(coverage_row('Q2',f'{4*g:02d}-{4*g+4:02d}',e2[:,g*24:g*24+24],b2[:,g*24:g*24+24],scope='4h_group'))
    month=np.array([d.isoformat()[:7] for d in data.dates[31:]])
    for key in sorted(set(month)):
        ix=month==key
        cov.append(coverage_row('Q2',key,e2[ix],b2[ix],scope='month'))
        cov.append(coverage_row('Q3',key,er3[ix],mr3[ix],scope='month'))
    save_csv(out/'coverage.csv',pd.DataFrame(cov))
    np.savez_compressed(out/'forecast_evidence.npz',q2_actual_net=data.net_load_kwh[31:],
       q2_actual_load=data.load_kw[31:],q2_actual_pv=data.pv_kw[31:],
       **{f'{mode}_load':fs[mode].load_kw[31:] for mode in modes},
       **{f'{mode}_pv':fs[mode].pv_kw[31:] for mode in modes},
       **{f'{mode}_margin':bs[mode][31:] for mode in modes},
       q3_forecast_latest=fr3,q3_error_latest=er3,q3_margin_latest=mr3,
       q3_all_errors=a3.errors_kwh[31:],q3_all_margins=b3[31:])

    for gamma in [0.,.5,1.]:
        print(f'Q3 gamma {gamma}',flush=True)
        r=m.run_policy(d3,a3,b3,m.PolicyConfig(f'gamma_{gamma}','diagnostic',gamma=gamma,collect_ledger=True))
        assert r.validation['hard_checks_pass'],r.validation
        ev=r.ledger[r.ledger.date>='2025-02-01'];daily=r.daily[r.daily.date>='2025-02-01']
        gamma_rows.append(dict(question='Q3',gamma=gamma,actual_total_cost_yuan=r.metrics['actual_total_cost_yuan'],
           emergency_kwh=r.metrics['emergency_purchase_kwh'],initial_soc_feb1_kwh=float(ev.soc_start_kwh.iloc[0]),
           terminal_soc_kwh=float(ev.soc_end_kwh.iloc[-1]),mean_carryover_soc_kwh=float(ev.groupby('date').soc_end_kwh.last().iloc[:-1].mean()),
           upper_carryover_days=int((ev.groupby('date').soc_end_kwh.last().iloc[:-1]>=10800-1e-4).sum())))
        validations[f'Q3_gamma_{gamma}']=r.validation
        ev.to_csv(out/f'q3_gamma_{gamma}_interval.csv.gz',index=False,encoding='utf-8-sig',float_format='%.10f',compression='gzip')
        save_csv(out/f'q3_gamma_{gamma}_daily.csv',r.daily)
        if gamma==.5:
            cs=['original_contract_kwh','final_contract_kwh','actual_charge_kwh','actual_discharge_kwh','actual_emergency_kwh','soc_end_kwh','actual_total_cost_yuan']
            reg['q3_baseline_ledger']=float(np.max(abs(ev[cs].to_numpy()-old3[cs].to_numpy())))
            assert reg['q3_baseline_ledger']<5e-5
        print(json.dumps(gamma_rows[-1]),flush=True)
    gamma_df=pd.DataFrame(gamma_rows)
    gamma_df['cost_change_vs_gamma05_yuan']=gamma_df.apply(lambda row:row.actual_total_cost_yuan-gamma_df[(gamma_df.question==row.question)&(gamma_df.gamma==.5)].actual_total_cost_yuan.iloc[0],axis=1)
    save_csv(out/'gamma_comparison.csv',gamma_df)
    save_json(out/'engine_verification.json',{'status':'PASS','regression':reg,'strategy_checks':validations})

    print('Checking future-observation perturbations of all three Q2 predictors...',flush=True)
    causal=[]
    for cut in [31,180,300]:
        load=data.load_kw.copy();pv=data.pv_kw.copy();load[cut:]+=1234.;pv[cut:]*=.3
        alt=replace(data,load_kw=load,pv_kw=pv,net_load_kwh=(load-pv)/6)
        af=forecast(alt,'weekday')
        for mode in modes:
            altered=baseline_forecast(alt,mode,af);mb=c.build_risk_margins(altered,.8)
            diff=float(max(np.nanmax(abs(fs[mode].net_load_kwh[1:cut+1]-altered.net_load_kwh[1:cut+1])),
                           np.max(abs(bs[mode][:cut+1]-mb[:cut+1]))))
            assert diff<1e-9,(cut,mode,diff)
            causal.append(dict(cut_day_index=cut,mode=mode,max_difference=diff))
    assert all(c.sha256(PROJECT/p)==h for p,h in hashes.items())
    summary=dict(status='CONDITIONAL',verification_status='PASS',official_strategy_changed=False,
       forecast_comparison=forecast_rows,coverage_overall=[r for r in cov if r['group']=='overall'],
       gamma_comparison=gamma_df.to_dict('records'),causality_checks=causal,
       source_hashes_unchanged=True,runtime_seconds=time.perf_counter()-started,
       limitations=protocol['limitations']+['Empirical coverage is not a formal chance constraint or guaranteed reliability.',
         'Gamma grid is a sensitivity diagnostic, not an optimizer; policy evaluation includes different January-induced February starting SOC.'])
    save_json(out/'summary.json',summary)
    print('COMPLETE '+str(out/'summary.json'),flush=True)


if __name__=='__main__':main()
