"""Same-contract causal storage-control diagnostic; never changes official Q3 outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = ROOT
sys.path.insert(0, str(DEFAULT_PROJECT / 'code'))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

ETA = 0.9
LOW, HIGH, U = 1200., 10800., 5000 / 6
TOL = 5e-5
PRIMARY_TOL = 1e-7


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@lru_cache(None)
def recurrence(n):
    a = lil_matrix((n, 3 * n), dtype=float)
    for i in range(n):
        a[i, i] = -ETA
        a[i, n+i] = 1 / ETA
        a[i, 2*n+i] = 1
        if i: a[i, 2*n+i-1] = -1
    return a.tocsr()


def choose_discharge(gaps, prices, soc, terminal_value, force_lp=False):
    """gaps[0] is observed current gap; all later gaps are causal forecasts.

    Only surplus contract/PV can charge. Emergency energy cannot charge storage.
    Primary: forecast emergency expense minus inherited terminal value.
    Tie-break: maximize current discharge among primary-optimal plans.
    Returns current discharge and whether an LP pair was used.
    """
    available = max(0., ETA * (soc-LOW))
    greedy = min(max(0., gaps[0]), U, available)
    if greedy <= 1e-9: return 0., False
    if not force_lp and (len(gaps) == 1 or prices[0] >= np.max(prices[1:]) - 1e-12):
        return greedy, False
    n = len(gaps)
    bounds = [(0., min(U, max(0., -float(z)))) for z in gaps]
    bounds += [(0., min(U, max(0., float(z)))) for z in gaps]
    bounds += [(LOW, HIGH)] * n
    rhs = np.zeros(n); rhs[0] = soc
    objective = np.r_[np.zeros(n), -5 * prices, np.zeros(n)]
    objective[-1] = -terminal_value
    options = {'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8}
    first = linprog(objective, A_eq=recurrence(n), b_eq=rhs, bounds=bounds,
                    method='highs', options=options)
    if not first.success: raise RuntimeError(f'storage LP failed: {first.message}')
    secondary = np.zeros(3*n); secondary[n] = -1
    second = linprog(secondary, A_eq=recurrence(n), b_eq=rhs, bounds=bounds,
                     A_ub=csr_matrix(objective.reshape(1,-1)),
                     b_ub=[first.fun+PRIMARY_TOL], method='highs', options=options)
    if not second.success: raise RuntimeError(f'tie-break LP failed: {second.message}')
    if objective @ second.x > first.fun + 2e-6: raise AssertionError('objective tolerance')
    return float(np.clip(second.x[n], 0., greedy)), True


def unit_checks():
    # One deliverable kWh. Saving it for the second, dearer shortage costs 5 vs 10.
    d, _ = choose_discharge(np.array([1.,1.]), np.array([1.,2.]), LOW+1/ETA, 0.)
    assert d < 1e-6, d
    flat, _ = choose_discharge(np.array([1.,1.]), np.array([1.,1.]), LOW+1/ETA, 0., True)
    assert abs(flat-1) < 1e-6, flat
    # Free replenishment before an expensive shortage removes the need to reserve.
    refill, _ = choose_discharge(np.array([1.,-2.,1.]), np.array([1.,1.,2.]), LOW+1/ETA, 0.)
    assert abs(refill-1) < 1e-6, refill
    return {'two_period_reserve': d, 'flat_price_tie_discharge': flat,
            'replenishment_before_peak_discharge': refill, 'status': 'PASS'}


def prepare(project, out):
    print('Loading original Q3 inputs and reconstructing causal forecast/margin archive...', flush=True)
    source_code = project / 'code/q3_model.py'
    spec = importlib.util.spec_from_file_location('q3_diagnostic_source', source_code)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    pack = project / 'comparison'
    ledger_path = pack / 'q3_main_interval_ledger.csv.gz'
    events_path = pack / 'q3_adjustment_events.csv.gz'
    old = pd.read_csv(ledger_path).sort_values(['date','interval_index']).reset_index(drop=True)
    events = pd.read_csv(events_path)
    dates = old.date.unique()
    assert len(old) == 334 * 144 and len(dates) == 334 and len(events) == 72144
    data = m.load_data()
    archive = m.build_forecast_archive(data, 'official')
    margins = m.build_risk_margins(archive, .8)
    print('Forecasts ready; checking 72,144 contract revision records...', flush=True)
    snapshots = np.full((334,4,144), np.nan)
    max_prev = 0.; max_fee = 0.; max_forecast = 0.; max_margin = 0.
    for d, day in enumerate(dates):
        rows = old.iloc[d*144:(d+1)*144]
        q = rows.original_contract_kwh.to_numpy().copy()
        snapshots[d,0] = q
        fee = np.zeros(144)
        for k, hour in enumerate([6,12,18], start=1):
            cur = events[(events.date == day) & (events.release_hour == hour)].sort_values('interval_index')
            idx = cur.interval_index.to_numpy(int)-1
            assert np.array_equal(idx, np.arange(hour*6,144))
            max_prev = max(max_prev, float(np.max(abs(q[idx]-cur.previous_contract_kwh))))
            new = cur.new_contract_kwh.to_numpy()
            cost = 1.5*data.price[idx]*np.maximum(new-q[idx],0)-.5*data.price[idx]*np.maximum(q[idx]-new,0)
            max_fee = max(max_fee, float(np.max(abs(cost-cur.incremental_adjustment_fee_yuan))))
            fee[idx] += cost; q[idx] = new; snapshots[d,k] = q
        assert np.max(abs(q-rows.final_contract_kwh.to_numpy())) < TOL
        assert np.max(abs(fee-rows.cumulative_adjustment_fee_yuan.to_numpy())) < TOL
        for k in range(4):
            sl = slice(k*36,(k+1)*36)
            max_forecast = max(max_forecast, float(np.max(abs(archive.net_kwh[d+31,k,sl]-rows.forecast_net_load_kwh.to_numpy()[sl]))))
            max_margin = max(max_margin, float(np.max(abs(margins[d+31,k,sl]-rows.risk_margin_kwh.to_numpy()[sl]))))
    assert max(max_prev,max_fee,max_forecast,max_margin) < TOL
    assert np.max(abs(data.net_load_kwh[31:].ravel()-old.actual_net_load_kwh.to_numpy())) < TOL
    hashes = {p.resolve().relative_to(project).as_posix(): m.sha256(p) for p in [
              source_code, ledger_path, events_path, m.PRICE_PATH,m.ACTUAL_PATH,m.FORECAST_PATH,Path(__file__)]}
    protocol = {
      'claim': 'Does price-aware forecast-based storage reservation improve greedy feedback under the same Q3 contracts?',
      'status': 'EXPERIMENT_ONLY_NOT_OFFICIAL_REPLACEMENT',
      'evaluation': ['2025-02-01','2025-12-31'], 'days':334,
      'arms': ['greedy_replay','price_aware_mpc'], 'alpha': .8, 'gamma': .5,
      'horizon': 'current 10-minute interval to current midnight; replan each interval with a deficit and higher future price',
      'current_information': 'same idealized current-interval net-load observation as paper Eq.9',
      'future_information': 'most recent released official forecast plus inherited risk margin; only then-effective contract snapshot; no anticipated revisions',
      'contracts': 'exogenous replay of original schedule, unveiled at 00/06/12/18; no endogenous next-day or intraday procurement changes',
      'primary': 'minimize forecast emergency cost minus gamma*eta*median(price)*terminal_SOC; last day terminal value zero',
      'tie_break': 'maximize current discharge within 1e-7 yuan primary tolerance',
      'actions': 'charge only actual surplus, discharge only deficit; no emergency charging; surplus charging stays greedy',
      'initial_soc_kwh':float(old.soc_start_kwh.iloc[0]), 'cross_day':'each arm carries its own actual SOC; no daily reset',
      'cost': 'identical original contract plus all original adjustment costs, plus new actual emergency expenses; no fictitious terminal credit',
      'predeclared_comparison': 'report signed cost difference without selecting parameters based on evaluation results',
      'verification_tolerance_kwh':TOL, 'input_hashes':hashes,
      'source_checks':{'contract_previous_max_abs':max_prev,'fee_max_abs':max_fee,
                       'forecast_max_abs':max_forecast,'margin_max_abs':max_margin},
      'unit_checks':unit_checks(),
      'limitation':'fixed-contract counterfactual isolates execution; does not measure fully reoptimized end-to-end policy or global optimality'
    }
    write_json(out/'protocol.json',protocol)
    return m, data, archive, margins, old, snapshots, protocol


def replay(old, snapshots, archive, margins, prices, arm, ndays, out):
    soc = float(old.soc_start_kwh.iloc[0]); result=[]; lp_count=0
    started=time.perf_counter()
    for d in range(ndays):
        day = str(old.date.iloc[d*144]); terminal=0. if d+31==364 else .5*ETA*float(np.median(prices))
        block=old.iloc[d*144:(d+1)*144]
        actual=block.actual_net_load_kwh.to_numpy()
        for t in range(144):
            k=t//36; known=snapshots[d,k]; gap=float(actual[t]-known[t]); s0=soc
            c=min(max(-gap,0.),U,max((HIGH-soc)/ETA,0.))
            greedy=min(max(gap,0.),U,max(ETA*(soc-LOW),0.))
            r=greedy
            if arm=='price_aware_mpc' and greedy>1e-9:
                predicted=archive.net_kwh[d+31,k,t:] + margins[d+31,k,t:] - known[t:]
                predicted=predicted.copy(); predicted[0]=gap
                r, used=choose_discharge(predicted,prices[t:],soc,terminal)
                lp_count+=int(used)
            e=max(gap-r,0.); w=max(-gap-c,0.)
            soc += ETA*c-r/ETA
            if not LOW-TOL<=soc<=HIGH+TOL: raise AssertionError(('SOC',day,t,soc))
            result.append((day,t+1,known[t],c,r,e,w,s0,soc,greedy-r,k*6))
        if (d+1)%20==0 or d+1==ndays:
            print(f'{arm}: {d+1}/{ndays} days, {lp_count} LP pairs, {time.perf_counter()-started:.1f}s',flush=True)
    r=pd.DataFrame(result,columns=['date','interval_index','contract_kwh','charge_kwh','discharge_kwh',
                    'emergency_kwh','unused_kwh','soc_start_kwh','soc_end_kwh','withheld_vs_greedy_kwh','available_release_hour'])
    ref=old.iloc[:len(r)]
    r['actual_net_kwh']=ref.actual_net_load_kwh.to_numpy()
    r['price_yuan_per_kwh']=ref.price_yuan_per_kwh.to_numpy()
    r['contract_cost_yuan']=ref.contract_total_cost_yuan.to_numpy()
    r['emergency_cost_yuan']=5*r.price_yuan_per_kwh*r.emergency_kwh
    r['actual_total_cost_yuan']=r.contract_cost_yuan+r.emergency_cost_yuan
    # Independent vectorized arithmetic verification of executed, not planned, values.
    checks={
      'balance':float(np.max(abs(r.contract_kwh+r.emergency_kwh+r.discharge_kwh-r.actual_net_kwh-r.charge_kwh-r.unused_kwh))),
      'state':float(np.max(abs(r.soc_end_kwh-r.soc_start_kwh-ETA*r.charge_kwh+r.discharge_kwh/ETA))),
      'chain':float(np.max(abs(r.soc_start_kwh.to_numpy()[1:]-r.soc_end_kwh.to_numpy()[:-1]))),
      'bounds':float(max(0,LOW-r.soc_end_kwh.min(),r.soc_end_kwh.max()-HIGH,r.charge_kwh.max()-U,r.discharge_kwh.max()-U)),
      'overlap':float(np.minimum(r.charge_kwh,r.discharge_kwh).max()),
      'contract':float(np.max(abs(r.contract_kwh-ref.final_contract_kwh.to_numpy())))
    }
    if arm=='greedy_replay':
        checks['baseline_reproduction']=float(np.max(abs(r[['charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','soc_start_kwh','soc_end_kwh']].to_numpy()-ref[['actual_charge_kwh','actual_discharge_kwh','actual_emergency_kwh','actual_unused_kwh','soc_start_kwh','soc_end_kwh']].to_numpy())))
    assert max(checks.values()) < TOL, checks
    cols=['contract_cost_yuan','emergency_cost_yuan','actual_total_cost_yuan','emergency_kwh','unused_kwh','charge_kwh','discharge_kwh','withheld_vs_greedy_kwh']
    daily=r.groupby('date')[cols].sum()
    daily['soc_start_kwh']=r.groupby('date').soc_start_kwh.first()
    daily['soc_end_kwh']=r.groupby('date').soc_end_kwh.last()
    r.to_csv(out/f'{arm}_interval.csv.gz',index=False,encoding='utf-8-sig',float_format='%.10f',compression='gzip')
    daily.to_csv(out/f'{arm}_daily.csv',encoding='utf-8-sig',float_format='%.10f')
    summary={x:float(r[x].sum()) for x in cols}
    summary.update(arm=arm,days=ndays,initial_soc_kwh=float(r.soc_start_kwh.iloc[0]),
        terminal_soc_kwh=float(r.soc_end_kwh.iloc[-1]),emergency_intervals=int((r.emergency_kwh>1e-6).sum()),
        withheld_intervals=int((r.withheld_vs_greedy_kwh>1e-6).sum()),lp_pairs=lp_count,
        runtime_seconds=time.perf_counter()-started,verification=checks)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary,daily,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--days',type=int,default=334)
    args=ap.parse_args()
    args.project=args.project.resolve()
    if not 1 <= args.days <= 334:ap.error('--days must be between 1 and 334')
    args.output.mkdir(parents=True,exist_ok=False)
    m,data,archive,margins,old,snapshots,protocol=prepare(args.project,args.output)
    summaries=[];daily_results=[]
    for arm in ['greedy_replay','price_aware_mpc']:
        sm,dr,ir=replay(old,snapshots,archive,margins,data.price,arm,args.days,args.output)
        summaries.append(sm);daily_results.append(dr)
    base,new=summaries
    saving=base['actual_total_cost_yuan']-new['actual_total_cost_yuan']
    comparison=pd.DataFrame(summaries).drop(columns='verification')
    comparison['saving_vs_greedy_yuan']=base['actual_total_cost_yuan']-comparison.actual_total_cost_yuan
    comparison.to_csv(args.output/'comparison.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    paired=pd.DataFrame({'greedy_cost_yuan':daily_results[0].actual_total_cost_yuan,'mpc_cost_yuan':daily_results[1].actual_total_cost_yuan})
    paired['saving_yuan']=paired.greedy_cost_yuan-paired.mpc_cost_yuan
    paired.to_csv(args.output/'paired_daily.csv',encoding='utf-8-sig',float_format='%.10f')
    paired.groupby(paired.index.str[:7]).sum().to_csv(args.output/'monthly.csv',encoding='utf-8-sig',float_format='%.10f')
    unchanged=all(m.sha256(args.project/path)==h for path,h in protocol['input_hashes'].items())
    assert unchanged
    summary={'status':'CONDITIONAL','verification_status':'PASS','official_strategy_changed':False,
      'claim':protocol['claim'],'arms':summaries,'saving_yuan':saving,
      'saving_percent':100*saving/base['actual_total_cost_yuan'],
      'days_mpc_cheaper':int((paired.saving_yuan>1e-4).sum()),
      'days_mpc_dearer':int((paired.saving_yuan< -1e-4).sum()),
      'max_daily_loss_yuan':float(max(0,-paired.saving_yuan.min())),
      'terminal_soc_difference_kwh':new['terminal_soc_kwh']-base['terminal_soc_kwh'],
      'source_hashes_unchanged':unchanged,'limitations':[protocol['limitation'],
        'risk level and terminal value inherited, not newly tuned; the test cannot establish superiority over all causal controllers',
        'current interval observation follows the same idealization as the original paper']}
    write_json(args.output/'summary.json',summary)
    print('COMPLETE '+str(args.output/'summary.json'),flush=True)


if __name__=='__main__': main()
