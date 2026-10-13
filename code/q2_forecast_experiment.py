"""Forecast-only diagnostic; never overwrites the approved M2 outputs.

Fixed before execution: 28-day load window, exp(-age/7) weights; PV unchanged.
weekday: weighted same-weekday mean. trend: weighted same-weekday linear
regression, extrapolated to today, nonnegative; mean fallback with <3 samples.
No parameter search. All variants rebuild past-error margins with the SAME rule.
"""
from dataclasses import replace
import json
import time
from pathlib import Path

import q2_common as c
import numpy as np
import pandas as pd
from q2_full_feedback import run_full_feedback_strategy, MAIN_CONFIG, save_frame, write_json

OUT = c.PROJECT_ROOT / 'results/Q2/experiments/round4_forecast_only'
OLD = c.PROJECT_ROOT / 'results/Q2/experiments/round3_full_feedback'
MODES = ('current', 'weekday', 'trend')


def forecast(data, mode):
    base = c.build_forecasts(data, 'exponential')
    if mode == 'current':
        return base
    load = base.load_kw.copy()
    for d in range(1, c.N_DAYS):
        ix = np.array([j for j in range(max(0, d-28), d)
                       if data.dates[j].weekday() == data.dates[d].weekday()], dtype=int)
        if not len(ix):
            continue  # same cold-start fallback as current forecast
        x = (ix-d).astype(float)
        w = np.exp(x/7)
        y = data.load_kw[ix]
        mean = np.average(y, axis=0, weights=w)
        if mode == 'trend' and len(ix) >= 3:
            xm = np.average(x, weights=w)
            slope = np.sum(w[:, None]*(x-xm)[:, None]*(y-mean), axis=0) / np.sum(w*(x-xm)**2)
            mean = mean - xm*slope
        load[d] = np.maximum(mean, 0)
    net = (load-base.pv_kw)*c.DELTA_HOURS
    return replace(base, load_kw=load, net_load_kwh=net,
                   errors_kwh=data.net_load_kwh-net)


def audit(ledger, daily, data):
    """Recompute ledger physics and money independently of run metrics."""
    a = lambda key: ledger[key].to_numpy(float)
    checks = {}
    def eq(name, residual):
        checks[name] = float(np.max(np.abs(residual)))
    q,e,ch,dis,w = [a(k) for k in ('planned_purchase_kwh','emergency_purchase_kwh',
                       'actual_charge_kwh','actual_discharge_kwh','actual_unused_kwh')]
    s,t,p = a('soc_start_kwh'),a('soc_end_kwh'),a('price_yuan_per_kwh')
    eq('raw_net_mapping', a('actual_net_load_kwh')-data.net_load_kwh.ravel())
    eq('raw_price_mapping', p-np.tile(data.price,c.N_DAYS))
    eq('balance', q+e+dis-a('actual_net_load_kwh')-ch-w)
    eq('soc_recurrence', t-s-.9*ch+dis/.9)
    eq('soc_chain', s[1:]-t[:-1])
    eq('initial_soc', s[:1]-6000)
    eq('contract_cost', a('contract_cost_yuan')-p*q)
    eq('emergency_cost', a('emergency_cost_yuan')-5*p*e)
    eq('charge_discharge_overlap', np.minimum(ch,dis))
    eq('emergency_unused_overlap', np.minimum(e,w))
    checks['bounds'] = float(max(0,1200-t.min(),t.max()-10800,ch.max()-5000/6,
                                 dis.max()-5000/6,-min(q.min(),e.min(),ch.min(),dis.min(),w.min())))
    for k in ('contract_cost_yuan','emergency_cost_yuan'):
        eq('daily_'+k, ledger.groupby('date',sort=True)[k].sum().to_numpy()-daily[k].to_numpy())
    eq('daily_total', daily.actual_cost_yuan-daily.contract_cost_yuan-daily.emergency_cost_yuan)
    eq('planner_actual_initial', daily.planner_initial_soc_kwh-daily.initial_soc_kwh)
    expected = [(d.isoformat(),t+1) for d in data.dates for t in range(c.N_INTERVALS)]
    shape_ok = list(zip(ledger.date,ledger.interval_index)) == expected
    return {'checks':checks,'coverage_pass':shape_ok,
            'all_pass':bool(shape_ok and all(v<=c.CHECK_TOLERANCE for v in checks.values()))}


def main():
    start=time.perf_counter()
    data=c.load_q2_data()
    sources = {str(p):c.sha256(p) for p in (c.PRICE_PATH,c.ACTUAL_PATH,Path(c.__file__),
               Path(__file__),Path(__file__).with_name('q2_full_feedback.py'),
               OLD/'tables/q2_M2_interval_ledger.csv', OLD/'tables/q2_M2_daily_summary.csv')}
    old_l=pd.read_csv(OLD/'tables/q2_M2_interval_ledger.csv')
    old_d=pd.read_csv(OLD/'tables/q2_M2_daily_summary.csv')
    checks={'saved_baseline_audit':audit(old_l,old_d,data)}
    assert checks['saved_baseline_audit']['all_pass'], checks
    old_metrics=json.loads((OLD/'metrics/M2_main_metrics.json').read_text(encoding='utf-8'))
    rows=[]; monthly=[]; validations={}
    for mode in MODES:
        print('Running '+mode,flush=True)
        f=forecast(data,mode)
        margin=c.build_risk_margins(f,.8)
        # Change target-day and all future observations. Forecast and margin
        # at the decision boundary must remain identical.
        causal=[]
        for cut in (7,31,180,300):
            load=data.load_kw.copy(); pv=data.pv_kw.copy()
            load[cut:]+=1234; pv[cut:]*=.3
            alt=replace(data,load_kw=load,pv_kw=pv,net_load_kwh=(load-pv)*c.DELTA_HOURS)
            fa=forecast(alt,mode); ma=c.build_risk_margins(fa,.8)
            causal.append(float(max(np.nanmax(abs(f.net_load_kwh[1:cut+1]-fa.net_load_kwh[1:cut+1])),
                                    np.max(abs(margin[:cut+1]-ma[:cut+1])))))
        assert max(causal)==0
        run=run_full_feedback_strategy(data,f,margin,replace(MAIN_CONFIG,strategy_id='E1_'+mode,role='diagnostic'))
        validations[mode]={'engine':run.validation,'independent_ledger':audit(run.ledger,run.daily,data),
                           'future_mutation_max_abs':max(causal)}
        assert run.validation['all_pass'] and validations[mode]['independent_ledger']['all_pass']
        if mode=='current':
            numeric=['planned_purchase_kwh','actual_charge_kwh','actual_discharge_kwh',
                     'emergency_purchase_kwh','soc_end_kwh','contract_cost_yuan','emergency_cost_yuan']
            diff=float(np.max(abs(run.ledger[numeric].to_numpy()-old_l[numeric].to_numpy())))
            checks['baseline_rerun_max_abs']=diff
            checks['baseline_total_difference_yuan']=run.metrics['actual_total_cost_yuan']-old_metrics['actual_total_cost_yuan']
            assert diff<c.CHECK_TOLERANCE and abs(checks['baseline_total_difference_yuan'])<.01
        err=f.errors_kwh[31:]*6
        high=data.price>=np.quantile(data.price,.75)
        m=dict(run.metrics)
        m.update(mode=mode,load_mae_kw=float(np.mean(abs(data.load_kw[31:]-f.load_kw[31:]))),
                 pv_mae_kw=float(np.mean(abs(data.pv_kw[31:]-f.pv_kw[31:]))),
                 net_mae_kw=float(np.mean(abs(err))),net_underestimate_p95_kw=float(np.quantile(err,.95)),
                 high_price_underestimate_mean_kw=float(np.maximum(err[:,high],0).mean()),
                 risk_margin_kwh=float(margin[31:].sum()),
                 savings_yuan=old_metrics['actual_total_cost_yuan']-run.metrics['actual_total_cost_yuan'])
        rows.append(m)
        d=run.daily[run.daily.date>='2025-02-01'].copy()
        d['month']=d.date.str[:7]
        cols=['contract_cost_yuan','emergency_cost_yuan','actual_cost_yuan','planned_purchase_kwh',
              'emergency_purchase_kwh','actual_unused_kwh','risk_margin_kwh']
        month=d.groupby('month')[cols].sum().reset_index();month['mode']=mode;monthly.append(month)
        save_frame(run.daily,OUT/f'tables/{mode}_daily.csv')
        # baseline ledger already exists; avoid duplicating it.
        if mode!='current': save_frame(run.ledger,OUT/f'tables/{mode}_interval.csv')
        write_json(OUT/f'metrics/{mode}.json',m)
        print(json.dumps({k:m[k] for k in ('mode','actual_total_cost_yuan','savings_yuan','net_mae_kw')},ensure_ascii=False),flush=True)
    save_frame(pd.DataFrame(rows),OUT/'tables/comparison.csv')
    save_frame(pd.concat(monthly),OUT/'tables/monthly.csv')
    checks['strategies']=validations
    checks['source_hashes_unchanged']=all(c.sha256(Path(p))==h for p,h in sources.items())
    assert checks['source_hashes_unchanged']
    write_json(OUT/'metrics/verification.json',checks)
    summary={'status':'completed','scope':'diagnostic_only_not_formal_replacement',
             'user_authorization':'那先做第一个实验（基准复核和只替换预测）',
             'parameters':'28-day exp(-age/7); same PV; alpha=.8 gamma=.5; no search',
             'input_and_code_hashes':sources,'environment':c.environment_info(),
             'runtime_seconds':time.perf_counter()-start,'metrics':rows,
             'limitations':['Retrospective structural comparison, not pristine holdout validation.',
                            'Margins recalculated by unchanged rule; includes induced margin/SOC changes.',
                            'No prediction of unseen-year generalization; no automatic main-method adoption.']}
    write_json(OUT/'run_summary.json',summary)
    write_json(c.PROJECT_ROOT/'robustness/Q2/forecast_experiment_summary.json',
               {'status':'CONDITIONAL','claim':'Forecast-only retrospective comparison',
                'evidence':str(OUT/'run_summary.json'),'physical_and_accounting':'PASS',
                'limitation':summary['limitations']})


if __name__=='__main__': main()
