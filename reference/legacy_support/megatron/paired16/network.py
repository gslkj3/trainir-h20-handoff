"""Select active RDMA ports mapped to the requested socket NIC on THIS host."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys


def select(sysroot, iface, mapping):
    net = sysroot/'class/net'/iface
    if not net.exists() or (net/'operstate').read_text().strip() != 'up':
        raise RuntimeError(f'{iface} missing or not UP')
    interfaces = {iface}
    slaves = net/'bonding/slaves'
    if slaves.exists(): interfaces.update(slaves.read_text().split())
    mapped = {}
    for line in mapping.splitlines():
        m = re.search(r'^(\S+)\s+port\s+(\d+)\s+==>\s+(\S+)\s+\(Up\)', line.strip())
        if m: mapped.setdefault((m[1],m[2]),set()).add(m[3])
    records = []
    for p in sorted((sysroot/'class/infiniband').glob('*/ports/*')):
        if not p.is_dir(): continue
        state = (p/'state').read_text().strip()
        names = set(mapped.get((p.parent.parent.name,p.name),set()))
        for f in (p/'gid_attrs/ndevs').glob('*'):
            try: names.add(f.read_text().strip())
            except OSError: pass
        records.append(dict(hca=p.parent.parent.name,port=p.name,state=state,
                            interfaces=sorted(names),link=(p/'link_layer').read_text().strip()))
    active = [r for r in records if r['state'].startswith('4:')]
    direct = [r for r in active if iface in r['interfaces']]
    chosen = direct or [r for r in active if interfaces.intersection(r['interfaces'])]
    if not chosen:
        raise RuntimeError('No ACTIVE RDMA port mapped to '+iface+': '+json.dumps(records))
    return '='+','.join(r['hca']+':'+r['port'] for r in chosen), records


def main():
    p=argparse.ArgumentParser();p.add_argument('--nodes',type=int,required=True);a=p.parse_args()
    if a.nodes == 1:
        print('unset NCCL_IB_HCA NCCL_NET')
        values=dict(NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='=lo',GLOO_SOCKET_IFNAME='lo')
    else:
        iface=os.environ.get('PAIR16_SOCKET_IFNAME','bond0')
        try:
            r=subprocess.run(['ibdev2netdev'],capture_output=True,text=True,timeout=15)
            mapping=r.stdout if r.returncode==0 else ''
        except (FileNotFoundError,subprocess.TimeoutExpired): mapping=''
        hca,records=select(Path('/sys'),iface,mapping)
        values=dict(NCCL_IB_HCA=hca,NCCL_IB_DISABLE='0',NCCL_NET='IB',
                    NCCL_SOCKET_IFNAME='='+iface,GLOO_SOCKET_IFNAME=iface)
        print('[NETWORK] '+json.dumps(dict(host=socket.gethostname(),selection=values,ports=records)),file=sys.stderr,flush=True)
    for k,v in values.items(): print('export '+k+'='+shlex.quote(v))


if __name__=='__main__':
    try: main()
    except Exception as e:
        print('[NETWORK ERROR] '+socket.gethostname()+': '+str(e),file=sys.stderr,flush=True)
        sys.exit(2)
