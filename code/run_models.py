"""四问的正式计算入口：模型函数取自现有程序，统一输入、结果导出和核对。

Q2明确采用 forecast(data, 'weekday')，不调用旧模块中的历史演示 main()。
每次计算写入独立目录；根目录的五份正式结果始终保持不变。
"""
from __future__ import annotations
import json
import time
from dataclasses import replace
from pathlib import Path

from export_results import export_q1,export_run,compare_workbooks

ROOT=Path(__file__).resolve().parents[1]


def save_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def save_frame(frame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10f')


def evaluation(frame):
    return frame[(frame['date']>='2025-02-01')&(frame['date']<='2025-12-31')].copy()


def check_run(run):
    valid=run.validation.get('hard_checks_pass',run.validation.get('all_pass',False))
    if not valid:
        raise RuntimeError(f'{run.config.strategy_id}的模型约束检查未通过：{run.validation}')
    if run.metrics.get('fallback_event_count',run.metrics.get('fallback_day_count',0)):
        raise RuntimeError(f'{run.config.strategy_id}触发求解回退，请检查求解环境；不将其作为正式重算结果')


def final_check(name,out,reports):
    result=compare_workbooks(ROOT/name,out/name)
    reports[name]=result
    print(f"  {name} 已导出；与既有结果最大逐格差异 {result['maximum_absolute_difference']:.3g}",flush=True)
    if result['status']!='PASS':
        save_json(out/'reproduction_check.json',reports)
        raise RuntimeError(f"{name}重算值与既有结果存在差异。新文件已单独保留；未覆盖正式结果。详情：{result['examples']}")


def run_q1(out,reports):
    import numpy as np
    import q1_common as c
    import q1_main
    import q1_baseline
    folder=out/'Q1';folder.mkdir(parents=True,exist_ok=True)
    data=c.load_q1_input();solved=q1_main.solve_main(data);main=solved.dispatch;base=q1_baseline.solve_baseline(data)
    previous=np.r_[c.SOC_INITIAL_KWH,main.soc_end[:-1]]
    balance=main.purchase+main.discharge-(data.load_kwh-data.pv_kwh)-main.charge-main.unused
    recurrence=main.soc_end-previous-.9*main.charge+main.discharge/.9
    maxima={'balance':float(np.max(abs(balance))),'storage':float(np.max(abs(recurrence))),
            'overlap':float(np.max(np.minimum(main.charge,main.discharge))),
            'soc_low':max(0.,float(1200-main.soc_end.min())),
            'soc_high':max(0.,float(main.soc_end.max()-10800)),
            'charge_limit':max(0.,float(main.charge.max()-5000/6)),
            'discharge_limit':max(0.,float(main.discharge.max()-5000/6)),
            'terminal':abs(float(main.soc_end[-1])-6000.)}
    if max(maxima.values())>1e-7:raise RuntimeError(f'Q1约束检查失败：{maxima}')
    c.write_csv(folder/'q1_interval_ledger.csv',
        ['interval_index','interval_label','price_yuan_per_kwh','actual_load_kw','pv_forecast_kw','purchase_kwh','charge_kwh','discharge_kwh','unused_kwh','soc_start_kwh','soc_end_kwh'],
        [[i+1,data.interval_labels[i],data.price[i],data.load_kw[i],data.pv_kw[i],main.purchase[i],main.charge[i],main.discharge[i],main.unused[i],previous[i],main.soc_end[i]] for i in range(144)])
    save_json(folder/'main_metrics.json',{'stage1_cost_yuan':solved.stage1_cost,'stage2_cost_yuan':solved.stage2_cost,
        'purchase_kwh':float(main.purchase.sum()),'charge_kwh':float(main.charge.sum()),'discharge_kwh':float(main.discharge.sum()),'checks':maxima})
    save_json(folder/'baseline_metrics.json',{'total_purchase_cost_yuan':base.cost,'purchase_kwh':float(base.purchase.sum()),'unused_kwh':float(base.unused.sum())})
    export_q1(ROOT,out,data,main);final_check('result1.xlsx',out,reports)


def run_q2(out,reports):
    import q2_common as c
    import q2_full_feedback as engine
    from q2_forecast_experiment import forecast,audit
    folder=out/'Q2';data=c.load_q2_data()
    f=forecast(data,'weekday');margins=c.build_risk_margins(f,.8)
    settings=[('main',replace(engine.MAIN_CONFIG,strategy_id='M3_weekday_pv7_alpha08_gamma05',role='main')),
              ('baseline',replace(engine.BASELINE_CONFIG,strategy_id='B3_same_forecast_no_margin'))]
    for role,config in settings:
        print(f'Q2 {role}：计算365天（正式统计334天）...',flush=True)
        run=engine.run_full_feedback_strategy(data,f,margins,config);check_run(run)
        independent=audit(run.ledger,run.daily,data)
        if not independent['all_pass']:raise RuntimeError(f'Q2独立检查失败：{independent}')
        save_frame(run.ledger,folder/f'q2_{role}_interval_ledger.csv')
        save_frame(run.daily,folder/f'q2_{role}_daily_summary.csv')
        save_json(folder/f'{role}_metrics.json',run.metrics)
        save_json(folder/f'{role}_validation.json',{'engine':run.validation,'independent':independent})
        print(f"  评价期费用 {run.metrics['actual_total_cost_yuan']:,.2f} 元",flush=True)
        if role=='main':
            export_run(ROOT,out,'result2.xlsx',run,c,is_q2=True)
            final_check('result2.xlsx',out,reports)


def run_q3(out,reports,full=False):
    import q3_model as m
    folder=out/'Q3';data=m.load_data();official=m.build_forecast_archive(data,'official')
    archives={'official':official}
    configs=[m.PolicyConfig('M1I_official_S3_alpha0.8_gamma0.5','main',collect_ledger=True),
             m.PolicyConfig('B1I_official_S3_b0_gamma0.5','usable_baseline',alpha=None)]
    if full:
        archives['historical']=m.build_forecast_archive(data,'historical')
        configs += [m.PolicyConfig('D_I_frequency_S0','diagnostic_frequency',selected_release_hours=(0,)),
                    m.PolicyConfig('D_I_frequency_S1','diagnostic_frequency',selected_release_hours=(0,6)),
                    m.PolicyConfig('D_I_frequency_S2','diagnostic_frequency',selected_release_hours=(0,6,12)),
                    m.PolicyConfig('D_I_no_adjustment','diagnostic_mechanism',allow_adjustment=False),
                    m.PolicyConfig('D_I_historical_pv','diagnostic_mechanism',forecast_source='historical'),
                    m.PolicyConfig('D_I_fixed_plan_execution','diagnostic_mechanism',feedback_execution=False),
                    m.PolicyConfig('S_I_alpha0.5','sensitivity',alpha=.5),
                    m.PolicyConfig('S_I_alpha0.95','sensitivity',alpha=.95),
                    m.PolicyConfig('S_I_gamma0','sensitivity',gamma=0.),
                    m.PolicyConfig('S_I_gamma1','sensitivity',gamma=1.)]
    cache={};runs=[]
    for config in configs:
        key=(config.forecast_source,config.alpha);archive=archives[config.forecast_source]
        if key not in cache:cache[key]=m.build_risk_margins(archive,config.alpha)
        print(f'Q3 {config.strategy_id}...',flush=True)
        run=m.run_policy(data,archive,cache[key],config);check_run(run);runs.append(run)
        print(f"  评价期费用 {run.metrics['actual_total_cost_yuan']:,.2f} 元",flush=True)
        if config.role in ('main','usable_baseline'):
            role='main' if config.role=='main' else 'baseline'
            save_frame(evaluation(run.daily),folder/f'q3_{role}_daily_summary.csv')
            save_json(folder/f'{role}_metrics.json',run.metrics);save_json(folder/f'{role}_validation.json',run.validation)
        if config.role=='main':
            save_frame(evaluation(run.ledger),folder/'q3_main_interval_ledger.csv')
            save_frame(evaluation(run.adjustments),folder/'q3_adjustment_events.csv')
            export_run(ROOT,out,'result3.xlsx',run,m,rolling=True)
            final_check('result3.xlsx',out,reports)
    comparison=m._comparison_frame(runs)
    save_frame(comparison,folder/'strategy_comparison.csv')
    if full:
        for role,name in [('diagnostic_frequency','frequency_comparison.csv'),('diagnostic_mechanism','mechanism_comparison.csv'),('sensitivity','sensitivity_comparison.csv')]:
            part=comparison[comparison['role']==role]
            save_frame(part,folder/name)
            _compare_frame(ROOT/'comparison'/name,part,reports)


def _compare_frame(reference,computed,reports):
    import pandas as pd
    old=pd.read_csv(reference).set_index('strategy_id');new=computed.set_index('strategy_id')
    columns=['actual_total_cost_yuan','emergency_purchase_kwh']
    maximum=0.
    for idx in old.index:
        if idx not in new.index:raise RuntimeError(f'缺少对照策略 {idx}')
        for col in columns:
            delta=abs(float(old.loc[idx,col])-float(new.loc[idx,col]));maximum=max(maximum,delta)
            if delta>5e-5:raise RuntimeError(f'{reference.name}/{idx}/{col}与存档结果不一致')
    reports[reference.name]={'status':'PASS','maximum_absolute_difference':maximum}


def run_q4(out,reports):
    import q4_model as m
    folder=out/'Q4';data=m.load_data();archives={k:m.build_forecast_archive(data,k) for k in ('historical','official')}
    for config in m._configurations('core'):
        archive=archives[config.forecast_source]
        margins=m.build_inherited_risk_margins(archive,config.question_variant,config.alpha)
        print(f'Q4 {config.strategy_id}...',flush=True)
        run=m.run_policy(data,archive,config,margins=margins);check_run(run)
        prefix=config.question_variant.lower().replace('-','')+('_main' if config.role=='main' else '_baseline')
        save_frame(evaluation(run.daily),folder/f'{prefix}_daily_summary.csv')
        save_json(folder/'metrics'/f'{prefix}_metrics.json',run.metrics)
        save_json(folder/'metrics'/f'{prefix}_validation.json',run.validation)
        print(f"  评价期费用 {run.metrics['actual_total_cost_yuan']:,.2f} 元",flush=True)
        if config.role=='main':
            save_frame(evaluation(run.ledger),folder/f'{prefix}_interval_ledger.csv')
            save_frame(evaluation(run.adjustments),folder/f'{prefix}_adjustment_events.csv')
            name='result4-2.xlsx' if config.question_variant=='Q4-2' else 'result4-3.xlsx'
            export_run(ROOT,out,name,run,m,rolling=config.allow_adjustment)
            final_check(name,out,reports)


def run_extra_q4(out,reports):
    import verify_inheritance as inheritance
    import run_common_soc_robustness as paired
    inheritance.OUTPUT_PATH=out/'Q4'/'inheritance_regression.json'
    print('Q4 恢复固定电价的一致性检验...',flush=True);inheritance.main()
    paired.ROUND_DIR=out/'Q4';paired.OUTPUT_PATH=out/'Q4'/'q4_robustness_summary.json'
    print('Q4 各分支内部相同初始储电量检验...',flush=True);paired.main()
    old=json.loads((ROOT/'comparison'/'q4_robustness_summary.json').read_text('utf-8'))
    new=json.loads(paired.OUTPUT_PATH.read_text('utf-8'));maximum=0.
    for variant in ('Q4-2','Q4-3'):
        for role in ('main','baseline'):
            for field in ('actual_total_cost_yuan','emergency_purchase_kwh'):
                delta=abs(old['variants'][variant][role][field]-new['variants'][variant][role][field]);maximum=max(maximum,delta)
                if delta>5e-5:raise RuntimeError(f'{variant}/{role}/{field}共同起点结果不一致')
    if new['status']!='PASS':raise RuntimeError('共同初始状态检验未通过')
    reports['q4_robustness_summary.json']={'status':'PASS','maximum_absolute_difference':maximum}
    reports['inheritance_regression.json']=json.loads(inheritance.OUTPUT_PATH.read_text('utf-8'))


def run(task:str,out:Path):
    out.mkdir(parents=True,exist_ok=False)
    reports={};start=time.time()
    tasks=['q1','q2','q3','q4'] if task in ('all','full') else [task]
    try:
        for key in tasks:
            print('\n'+key.upper()+' 开始计算',flush=True)
            if key=='q1':run_q1(out,reports)
            elif key=='q2':run_q2(out,reports)
            elif key=='q3':run_q3(out,reports,full=(task=='full'))
            elif key=='q4':run_q4(out,reports)
        if task=='full':run_extra_q4(out,reports)
    finally:
        save_json(out/'reproduction_check.json',{'task':task,'runtime_seconds':time.time()-start,'checks':reports})
    print(f'\n全部所选任务完成；本次输出：{out}',flush=True)
    return reports
