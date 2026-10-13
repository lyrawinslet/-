"""将本次计算结果写入 Excel；只使用 Python 标准库，不依赖 Node.js 或 Excel 软件。

根目录的既有结果文件仅提供工作表名称、表头与样式。导出时清空所有数据行，
用本次模型结果重新填入；写入后逐格读取核对。不会覆盖根目录的提交结果。
"""
from __future__ import annotations

import math
import posixpath
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
START = "2025-02-01"
END = "2025-12-31"


def column_name(index: int) -> str:
    text = ""
    while index:
        index, digit = divmod(index - 1, 26)
        text = chr(65 + digit) + text
    return text


def column_index(address: str) -> int:
    value = 0
    for ch in re.match(r"[A-Z]+", address).group():
        value = value * 26 + ord(ch) - 64
    return value


def excel_date(value) -> int:
    if isinstance(value, datetime):
        value = value.date()
    elif not isinstance(value, date):
        value = date.fromisoformat(str(value)[:10])
    return (value - date(1899, 12, 30)).days


def sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    book = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
    result = {}
    for node in book.find("s:sheets", NS):
        target = targets[node.attrib[f"{{{REL}}}id"]]
        result[node.attrib["name"]] = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
    return result


def read_workbook(path: Path) -> dict[str, dict[str, object]]:
    """Read literal/cached cell values without launching spreadsheet software."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"工作簿压缩结构损坏：{path.name}/{bad}")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        for title, member in sheet_paths(archive).items():
            cells = {}
            root = ET.fromstring(archive.read(member))
            for cell in root.findall("s:sheetData/s:row/s:c", NS):
                address = cell.attrib["r"]
                value = cell.find("s:v", NS)
                kind = cell.get("t")
                if kind == "e":
                    raise ValueError(f"{path.name}/{title}/{address}：单元格错误")
                if kind == "inlineStr":
                    content = cell.find("s:is", NS)
                    parsed = "".join(content.itertext()) if content is not None else ""
                elif value is None or value.text is None:
                    parsed = None
                elif kind == "s":
                    parsed = strings[int(value.text)]
                elif kind in ("str", "d"):
                    parsed = value.text
                elif kind == "b":
                    parsed = bool(int(value.text))
                else:
                    parsed = float(value.text)
                    if not math.isfinite(parsed):
                        raise ValueError(f"{path.name}/{title}/{address}：非有限数值")
                if parsed is not None:
                    cells[address] = parsed
            result[title] = cells
    return result


def _sheet_xml(original: bytes, rows: list[list[object]]) -> bytes:
    text = original.decode("utf-8")
    match = re.search(r"<(?P<p>(?:[\w]+:)?)sheetData\b[^>]*>.*?</(?P=p)sheetData>", text, re.S)
    if match is None:
        raise ValueError("缺少 sheetData，不能使用该工作表作为格式基准")
    prefix = match.group("p")
    root = ET.fromstring(original)
    original_rows = root.findall("s:sheetData/s:row", NS)
    header = original_rows[0]
    header_end = re.search(r"<" + prefix + r"row\b.*?</" + prefix + r"row>", match.group(), re.S)
    styles = {}
    for row in original_rows[1:]:
        for cell in row:
            styles.setdefault(column_index(cell.attrib["r"]), cell.get("s", "0"))
    chunks = [f"<{prefix}sheetData>", header_end.group()]
    width = max(column_index(c.attrib["r"]) for c in header)
    for row_index, values in enumerate(rows, 2):
        if len(values) != width:
            raise ValueError(f"输出第{row_index}行列数应为{width}，实际为{len(values)}")
        chunks.append(f'<{prefix}row r="{row_index}" ht="18" customHeight="1">')
        for col, value in enumerate(values, 1):
            address = column_name(col) + str(row_index)
            attrs = f'r="{address}" s="{styles.get(col, "0")}"'
            if value is None:
                chunks.append(f"<{prefix}c {attrs} />")
            elif isinstance(value, str):
                chunks.append(f'<{prefix}c {attrs} t="inlineStr"><{prefix}is><{prefix}t>{escape(value)}</{prefix}t></{prefix}is></{prefix}c>')
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{address}不能输出非有限值")
                # 仅去除浮点噪声；不把显示精度应用于模型计算。
                number = 0.0 if abs(number) < 1e-9 else number
                chunks.append(f'<{prefix}c {attrs} t="n"><{prefix}v>{number:.17g}</{prefix}v></{prefix}c>')
        chunks.append(f"</{prefix}row>")
    chunks.append(f"</{prefix}sheetData>")
    text = text[:match.start()] + "".join(chunks) + text[match.end():]
    # 原表没有数据公式、表对象或合并数据单元格；保留页边距、列宽和冻结设置。
    if root.find("s:mergeCells", NS) is not None or root.find("s:tableParts", NS) is not None:
        raise ValueError("格式基准含额外合并/表对象，需单独核对")
    dimension = f'<{prefix}dimension ref="A1:{column_name(width)}{len(rows)+1}" />'
    if re.search(r"<" + prefix + r"dimension\b[^>]*/>", text):
        text = re.sub(r"<" + prefix + r"dimension\b[^>]*/>", dimension, text, count=1)
    return text.encode("utf-8")


def write_workbook(template: Path, output: Path, tables: dict[str, list[list[object]]]) -> dict:
    if template.resolve() == output.resolve():
        raise ValueError("不能覆盖根目录正式结果；请选择独立输出目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".xlsx.tmp")
    try:
        with zipfile.ZipFile(template) as archive:
            paths = sheet_paths(archive)
            if set(paths) != set(tables):
                raise ValueError(f"{output.name}工作表集合不匹配")
            changed = {paths[name]: _sheet_xml(archive.read(paths[name]), rows) for name, rows in tables.items()}
            with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
                for info in archive.infolist():
                    target.writestr(info, changed.get(info.filename, archive.read(info.filename)))
        pending.replace(output)
    finally:
        if pending.exists():
            pending.unlink()
    # 核对写入后的值与本次计算值，而非仅检验文件能打开。
    actual = read_workbook(output)
    checked = 0
    for name, rows in tables.items():
        expected_addresses = {column_name(j)+str(i) for i,row in enumerate(rows,2) for j,v in enumerate(row,1) if v is not None}
        for i, row in enumerate(rows, 2):
            for j, value in enumerate(row, 1):
                address = column_name(j) + str(i)
                parsed = actual[name].get(address)
                if value is None:
                    if parsed is not None:
                        raise ValueError(f"{name}!{address}应为空")
                elif isinstance(value, str):
                    if parsed != value:
                        raise ValueError(f"{name}!{address}文本写入不一致")
                else:
                    if parsed is None or abs(float(parsed) - float(value)) > 2e-8:
                        raise ValueError(f"{name}!{address}数值写入不一致")
                    checked += 1
        stale = [a for a in actual[name] if int(re.search(r"\d+",a).group())>1 and a not in expected_addresses]
        if stale:
            raise ValueError(f"{name}含旧数据残留：{stale[:3]}")
    return {"numeric_cells_checked": checked, "status": "PASS"}


def export_q1(root: Path, output: Path, data, dispatch) -> dict:
    plan = [[label, float(amount)] for label, amount in zip(data.interval_labels, dispatch.purchase)]
    storage = []
    for block in range(6):
        start = block*24
        label = f"{block*4:02d}:00-{(block+1)*4:02d}:00"
        storage.append([label, float(dispatch.charge[start:start+24].sum()), float(dispatch.discharge[start:start+24].sum()),
                        "0:00" if block == 0 else "24:00" if block == 1 else None,
                        6000.0 if block == 0 else float(dispatch.soc_end[-1]) if block == 1 else None])
    return write_workbook(root/"result1.xlsx", output/"result1.xlsx", {"计划购电量":plan,"充放电量":storage})


def export_run(root: Path, output: Path, name: str, run, model, is_q2: bool = False, rolling: bool = False) -> dict:
    ledger = run.ledger
    if ledger is None:
        raise ValueError("缺少本次运行的逐时段结果，不能导出")
    ledger = ledger[(ledger["date"]>=START)&(ledger["date"]<=END)].copy().sort_values(["date","interval_index"])
    daily = run.daily[(run.daily["date"]>=START)&(run.daily["date"]<=END)].copy()
    expected_dates=[(date(2025,2,1)+timedelta(days=i)).isoformat() for i in range(334)]
    if len(ledger)!=48096 or daily["date"].tolist()!=expected_dates:
        raise ValueError("导出范围必须为2025年2—12月的334天")
    by_day=daily.set_index("date")
    purchase=[];adjusted=[];storage=[]
    for day,group in ledger.groupby("date",sort=True):
        ordered=group.sort_values("interval_index")
        if ordered["interval_index"].tolist()!=list(range(1,145)):
            raise ValueError(f"{day}的时段不完整")
        summary=by_day.loc[day]
        qfield="planned_purchase_kwh" if is_q2 else "original_contract_kwh"
        cfield="contract_cost_yuan" if is_q2 else "original_contract_cost_yuan"
        purchase.append([excel_date(day),*map(float,ordered[qfield]),float(summary[qfield]),float(summary[cfield])])
        if rolling:
            adjusted.append([excel_date(day),*map(float,ordered["final_contract_kwh"]),float(summary["final_contract_kwh"]),float(summary["actual_total_cost_yuan"])])
        initial=float(ordered.iloc[0]["soc_start_kwh"])
        terminal=float(ordered.iloc[-1]["soc_end_kwh"])
        for block in range(6):
            part=ordered.iloc[block*24:(block+1)*24]
            storage.append([excel_date(day) if block==0 else None,f"{block*4:02d}:00-{(block+1)*4:02d}:00",
                            float(part["actual_charge_kwh"].sum()),float(part["actual_discharge_kwh"].sum()),
                            "0:00" if block==0 else "24:00" if block==1 else None,
                            initial if block==0 else terminal if block==1 else None])
    events=model.emergency_events(ledger)
    emergency=[];prev=None
    for record in events.to_dict("records"):
        day=record["date"]
        emergency.append([excel_date(day) if day!=prev else None,record["emergency_time_range"],float(record["emergency_purchase_kwh"])])
        prev=day
    tables={"计划购电量":purchase}
    if rolling:tables["调整购电量"]=adjusted
    tables.update({"充放电量":storage,"紧急购电量":emergency})
    return write_workbook(root/name,output/name,tables)


def compare_workbooks(reference: Path, computed: Path, tolerance: float = 5e-5) -> dict:
    """Compare all data cells; header and style differences are not numeric evidence."""
    old=read_workbook(reference);new=read_workbook(computed)
    differences=[];count=0;maximum=0.0
    if set(old)!=set(new):raise ValueError("工作表集合不同")
    for name in old:
        a={k:v for k,v in old[name].items() if int(re.search(r"\d+",k).group())>1}
        b={k:v for k,v in new[name].items() if int(re.search(r"\d+",k).group())>1}
        for key in a.keys()|b.keys():
            x=a.get(key);y=b.get(key)
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                delta=abs(float(x)-float(y));maximum=max(maximum,delta);count+=1
                if delta>tolerance:differences.append((name,key,x,y))
            elif x!=y:differences.append((name,key,x,y))
    return {"file":reference.name,"numeric_cells_compared":count,"maximum_absolute_difference":maximum,
            "tolerance":tolerance,"different_cells":len(differences),"examples":differences[:8],
            "status":"PASS" if not differences else "DIFFERENT"}
