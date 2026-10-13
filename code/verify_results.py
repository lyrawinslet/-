"""从原始输入及现有结果重算费用、储能反馈与逐次调整；仅使用标准库。

python run.py --task verify
本程序不是重新求解优化模型；重算模型使用 --task all 或 --task full。
"""
from __future__ import annotations
import csv
import gzip
import math
from datetime import date,timedelta
from pathlib import Path
from export_results import read_workbook, column_name, excel_date

TOL = 5e-5
EXPECTED_COST = {"result1.xlsx":35126.94859028963,"result2.xlsx":13923903.386969224,
                 "result3.xlsx":13833843.322638258,"result4-2.xlsx":14571629.312229294,
                 "result4-3.xlsx":14487237.236657604}


def verify_package(root: Path) -> dict:
    checks=0;maximum=0.0
    def near(actual,expected,label,tol=TOL):
        nonlocal checks,maximum
        checks+=1
        delta=abs(float(actual)-float(expected));maximum=max(maximum,delta)
        if not math.isfinite(delta) or delta>tol:
            raise ValueError(f"核对失败 {label}: {actual} != {expected}；差值 {delta}")
    fixed=next(iter(read_workbook(root/'data'/'附件1.xlsx').values()))
    actual=read_workbook(root/'data'/'附件2.xlsx')
    floating=next(iter(read_workbook(root/'data'/'附件4.xlsx').values()))
    price0=[fixed[f'B{i+2}'] for i in range(144)]
    names=list(EXPECTED_COST);report={}
    for name in names:
        wb=read_workbook(root/name)
        plan=wb['计划购电量'];store=wb['充放电量']
        if name=='result1.xlsx':
            q=[plan[f'B{i+2}'] for i in range(144)]
            cost=math.fsum(p*v for p,v in zip(price0,q))
            near(cost,EXPECTED_COST[name],name+'总费')
            near(store['E2'],6000,name+'期初');near(store['E3'],6000,name+'期末')
            near(0.9*math.fsum(store[f'B{i}'] for i in range(2,8)),math.fsum(store[f'C{i}'] for i in range(2,8))/0.9,name+'首末能量')
            report[name]={'total_cost_yuan':cost,'intervals':144,'scope':'购电费及四小时储能汇总核对'}
            print(f'[PASS] {name}: {cost:,.2f} 元',flush=True)
            continue
        rolling='调整购电量' in wb
        variable=name.startswith('result4')
        final=wb['调整购电量'] if rolling else plan
        record_by_day={}
        if rolling:
            fname='q3_adjustment_events.csv.gz' if name=='result3.xlsx' else 'q43_main_adjustment_events.csv.gz'
            with gzip.open(root/'comparison'/fname,'rt',encoding='utf-8-sig',newline='') as handle:
                for row in csv.DictReader(handle):record_by_day.setdefault(row['date'],[]).append(row)
        all_expected_events=[];total=0.0;emergency_total=0.0;previous_end=None;contract_cost=0.0;adjustment_cost=0.0
        for d in range(334):
            day=date(2025,2,1)+timedelta(days=d);serial=excel_date(day);r=d+2;rawrow=d+33
            near(plan[f'A{r}'],serial,name+'日期')
            if rolling:near(final[f'A{r}'],serial,name+'调整页日期')
            p=[floating[f'{column_name(t+2)}{rawrow}'] if variable else price0[t] for t in range(144)]
            q=[plan[f'{column_name(t+2)}{r}'] for t in range(144)]
            y=[final[f'{column_name(t+2)}{r}'] for t in range(144)]
            near(plan[f'EP{r}'],math.fsum(q),name+'日初总电量')
            basecost=math.fsum(p[t]*q[t] for t in range(144));near(plan[f'EQ{r}'],basecost,name+'日初合同费')
            fees=[]
            if rolling:
                current=q.copy();events=record_by_day.get(day.isoformat(),[])
                expected_keys=[(hour,t) for hour in (6,12,18) for t in range(hour*6+1,145)]
                found_keys=[(int(row['release_hour']),int(row['interval_index'])) for row in events]
                if found_keys!=expected_keys:raise ValueError(f'{day}的合同调整记录顺序或范围不完整')
                for row in events:
                    t=int(row['interval_index'])-1;old=float(row['previous_contract_kwh']);new=float(row['new_contract_kwh'])
                    near(old,current[t],name+'上一合同');inc=max(new-old,0);dec=max(old-new,0)
                    near(float(row['increase_kwh']),inc,name+'增购量');near(float(row['decrease_kwh']),dec,name+'减购量')
                    fee=p[t]*(1.5*inc-.5*dec);near(float(row['incremental_adjustment_fee_yuan']),fee,name+'调整费')
                    current[t]=new;fees.append(fee)
                for t in range(144):near(current[t],y[t],name+'最终合同')
                near(final[f'EP{r}'],math.fsum(y),name+'最终总电量')
            startrow=2+d*6;S=float(store[f'F{startrow}']);initial=S
            near(store[f'A{startrow}'],serial,name+'储能页日期')
            if previous_end is not None:near(S,previous_end,name+'跨日状态')
            charges=[];discharges=[];emergencies=[]
            for t in range(144):
                col=column_name(t+2)
                n=(float(actual['小区负载'][f'{col}{rawrow}'])-float(actual['光伏发电实际功率'][f'{col}{rawrow}']))/6
                gap=n-y[t]
                charge=min(max(-gap,0.0),5000/6,max(0.0,(10800-S)/.9))
                discharge=min(max(gap,0.0),5000/6,max(0.0,.9*(S-1200)))
                emergency=max(gap+charge-discharge,0.0);S+=.9*charge-discharge/.9
                if not 1200-TOL<=S<=10800+TOL:raise ValueError(f'{name}储量越界')
                charges.append(charge);discharges.append(discharge);emergencies.append(emergency)
            for b in range(6):
                near(store[f'C{startrow+b}'],math.fsum(charges[b*24:(b+1)*24]),name+'四小时充电')
                near(store[f'D{startrow+b}'],math.fsum(discharges[b*24:(b+1)*24]),name+'四小时放电')
            near(store[f'F{startrow+1}'],S,name+'日末状态');previous_end=S
            dayemcost=math.fsum(5*p[t]*emergencies[t] for t in range(144))
            adj=math.fsum(fees);daycost=basecost+adj+dayemcost
            if rolling:near(final[f'EQ{r}'],daycost,name+'含紧急购电总费用')
            contract_cost+=basecost;adjustment_cost+=adj;total+=daycost;emergency_total+=math.fsum(emergencies)
            t=0
            while t<144:
                if emergencies[t]<=1e-6:t+=1;continue
                first=t
                while t+1<144 and emergencies[t+1]>1e-6:t+=1
                end=t+1
                interval=f'{first//6:02d}:{first%6*10:02d}-{end//6:02d}:{end%6*10:02d}'
                all_expected_events.append((serial,interval,math.fsum(emergencies[first:end])))
                t+=1
        # 确认只包含334天，而不是把1月合计混入结果。
        if f'A336' in plan:raise ValueError(f'{name}含评价期以外日期')
        ev=wb['紧急购电量'];actual_events=[];current_date=None
        for r in range(2,2+len(all_expected_events)):
            current_date=ev.get(f'A{r}',current_date)
            actual_events.append((current_date,ev.get(f'B{r}'),ev.get(f'C{r}')))
        if f'B{2+len(all_expected_events)}' in ev:raise ValueError(name+'紧急购电事件数量不符')
        for a,b in zip(actual_events,all_expected_events):
            near(a[0],b[0],name+'补购日期')
            if a[1]!=b[1]:raise ValueError(f'{name}补购区间不符：{a[1]} != {b[1]}')
            near(a[2],b[2],name+'补购事件电量')
        near(total,EXPECTED_COST[name],name+'评价期费用')
        report[name]={'intervals':48096,'total_cost_yuan':total,'original_contract_cost_yuan':contract_cost,
                      'adjustment_net_cost_yuan':adjustment_cost,'emergency_purchase_kwh':emergency_total,
                      'emergency_events':len(all_expected_events),'status':'PASS'}
        print(f'[PASS] {name}: {total:,.2f} 元；334天；紧急购电 {emergency_total:,.2f} kWh',flush=True)
    return {'status':'PASS','numeric_checks':checks,'maximum_absolute_difference':maximum,'workbooks':report}

if __name__=='__main__':
    print(verify_package(Path(__file__).resolve().parents[1]))
