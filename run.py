#!/usr/bin/env python3
"""微网购电与储能调度：跨平台统一入口（Python 3.11—3.13，64位）。

在 Windows、macOS 中均先解压整个文件夹，并保留内部相对位置。
Windows 可双击 start_windows.bat；macOS 可运行 bash start_mac.command。

命令行：
  python run.py --task verify    核对随包提交的五份结果；无需第三方依赖。
  python run.py --setup          在包内 .venv 安装计算依赖；首次需要网络。
  python run.py --task all       重算四问主方案及对应基准，导出五份 Excel。
  python run.py --task full      在 all 基础上复现预报时点、参数、机制和Q4检验。
  python run.py --task q2        只运行指定问题（支持 q1/q2/q3/q4）。

运行提示和依赖固定在本程序中，不另提供 requirements.txt 或 README。
根目录的结果文件作为提交值和格式基准保留，本次计算仅写入新建输出目录。
"""
from __future__ import annotations
import argparse
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parent
# 此组合已用于本包端到端重算。只安装公开Python依赖，不使用Node.js、Office或专有导表包。
PACKAGES=("numpy==2.3.5","scipy==1.17.0","pandas==2.2.3","openpyxl==3.1.5")


def console_setup():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    sys.dont_write_bytecode=True
    os.environ.setdefault('PYTHONUTF8','1')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    # 限制数值库线程，避免不同机器默认线程数量造成不必要的并行开销。
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'


def environment_python():
    return ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def setup_environment():
    import venv
    if not ((3,11)<=sys.version_info[:2]<=(3,13)) or sys.maxsize<=2**32:
        raise RuntimeError('请安装64位Python 3.11、3.12或3.13（建议3.12），再启动此程序。')
    exe=environment_python()
    if not exe.exists():
        print('正在本文件夹创建独立Python环境 .venv，不修改系统环境...',flush=True)
        venv.EnvBuilder(with_pip=True,clear=False).create(ROOT/'.venv')
    print('正在安装固定版本依赖；需要可连接Python软件源的网络...',flush=True)
    try:
        subprocess.run([str(exe),'-m','pip','install','--disable-pip-version-check','--only-binary=:all:',*PACKAGES],check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError('依赖安装未完成，请确认网络和Python软件源可用后重新执行 --setup。已提交的结果文件不受影响。') from error
    print('独立环境已准备完成。',flush=True)
    return exe


def has_dependencies(executable=None):
    if executable is not None:
        code='import importlib.metadata as m; '+ '; '.join(f"assert m.version({s.split('==')[0]!r})=={s.split('==')[1]!r}" for s in PACKAGES)
        return subprocess.run([str(executable),'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    try:
        return all(importlib.metadata.version(s.split('==')[0])==s.split('==')[1] for s in PACKAGES)
    except importlib.metadata.PackageNotFoundError:
        return False


def ensure_runtime(task,output):
    """Return None for current interpreter, otherwise invoke the isolated interpreter."""
    if has_dependencies():return None
    exe=environment_python()
    if not (exe.exists() and has_dependencies(exe)):
        if not sys.stdin.isatty():
            raise RuntimeError('计算依赖尚未准备，请先运行 python run.py --setup，再执行本次命令。')
        choice=input('缺少所需版本的计算依赖。是否在本文件夹创建独立环境并联网安装？[y/N] ').strip().lower()
        if choice not in ('y','yes'):raise RuntimeError('未安装依赖。可先选“核对现有结果”，或以后执行 --setup。')
        exe=setup_environment()
    if Path(sys.prefix).resolve()==(ROOT/'.venv').resolve():
        raise RuntimeError('独立环境依赖仍不完整，请运行 --setup 修复。')
    command=[str(exe),str(ROOT/'run.py'),'--task',task,'--worker']
    if output:command.extend(['--output',output])
    return subprocess.run(command,cwd=ROOT).returncode


def menu():
    print('\n微网购电与储能调度 — 辅助材料运行入口')
    print('1  核对现有五份结果（不重新求解，不需安装依赖）')
    print('2  重算四问主方案及对应基准，生成五份Excel')
    print('3  完整重算，同时运行参数、时点、机制和Q4对照检验')
    print('4  安装或修复包内独立计算环境（首次需联网）')
    print('5  核对两项补充实验的保存结果')
    print('6  重算两项补充实验并独立核验')
    print('0  退出')
    value=input('请选择 [0—6]：').strip()
    return {'0':'exit','1':'verify','2':'all','3':'full','4':'setup','5':'supplement_verify','6':'supplement'}.get(value,'invalid')


def main():
    console_setup()
    parser=argparse.ArgumentParser(description='五份结果核对、四问重算与Excel导出；不改写已提交结果。')
    parser.add_argument('--task',choices=('verify','q1','q2','q3','q4','all','full','supplement_verify','supplement'))
    parser.add_argument('--setup',action='store_true',help='联网安装固定版本依赖至包内.venv')
    parser.add_argument('--output',help='新建输出目录；相对路径以本程序所在文件夹为起点')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.setup:
        setup_environment();return 0
    task=args.task
    if task is None:
        if not sys.stdin.isatty():parser.print_help();return 0
        task=menu()
        if task=='exit':return 0
        if task=='invalid':raise RuntimeError('无效选项，请重新启动后选择0—6。')
        if task=='setup':setup_environment();return 0
    sys.path.insert(0,str(ROOT/'code'))
    if task=='verify':
        from verify_results import verify_package
        result=verify_package(ROOT)
        print(f"\n核对通过：{result['numeric_checks']:,}项数值核对；最大绝对差异 {result['maximum_absolute_difference']:.3g}。")
        return 0
    if not ((3,11)<=sys.version_info[:2]<=(3,13)) or sys.maxsize<=2**32:
        raise RuntimeError('重算需要64位Python 3.11—3.13（建议3.12）。')
    if task == 'supplement_verify':
        raise RuntimeError('本包提供补充实验汇总结果。逐时段数据的生成与核验请运行 python run.py --task supplement；五份结果核对请运行 python run.py --task verify。')
    runtime=ensure_runtime(task,args.output)
    if runtime is not None:return runtime
    output=Path(args.output).expanduser() if args.output else ROOT/'rerun'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not output.is_absolute():output=ROOT/output
    output=output.resolve()
    if output==ROOT or output.exists():raise RuntimeError('输出目录必须为尚不存在的新目录，避免覆盖现有材料。')
    print('运行环境：',platform.system(),sys.version.split()[0],flush=True)
    print('依赖版本：',', '.join(PACKAGES),flush=True)
    print('本次计算输出：',output,flush=True)
    if task in ('supplement_verify','supplement'):
        from run_supplement import run
    else:
        from run_models import run
    run(task,output)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        print(f'\n运行未完成：{error}',file=sys.stderr)
        print('已提交的原始结果未被覆盖。',file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\n运行已中止，原始结果未被覆盖。',file=sys.stderr)
        raise SystemExit(130)
