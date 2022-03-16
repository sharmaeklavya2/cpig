#!/usr/bin/env python3

"""Reads a set of conditional implications in CSV/JSON and outputs the hasse digraph."""

import sys
import argparse
import subprocess
import itertools
import os
import csv
import json
from typing import NamedTuple, Optional, Sequence


STDKEYS = {'from', 'to', 'description'}


def is_row_valid(row):
    return row.get('from') and row.get('to')


def csep_to_list(s):
    return s.strip().split(',') if s is not None else None


def cesep_to_dict(s):
    if s:
        entries = [kev.split('=', 1) for kev in (s or '').strip().split(',')]
        for a in entries:
            if len(a) == 1:
                a.append('1')
        return dict(entries)
    else:
        return {}


def infer_format(fpath):
    basename, ext = os.path.splitext(fpath)
    return ext[1:]


def read_file(ifpath, format):
    if format is None:
        format = infer_format(ifpath)
    if format == 'csv':
        rows = []
        with open(ifpath, newline='') as fp:
            reader = csv.DictReader(fp)
            for i, row in enumerate(reader):
                if is_row_valid(row):
                    rows.append(row)
                else:
                    raise ValueError('row {} is invalid in CSV file'.format(i))
        return rows
    elif format == 'json':
        with open(ifpath) as fp:
            try:
                return json.load(fp, parse_int=str)
            except json.JSONDecodeError as e:
                print('Invalid JSON in ' + ifpath, file=sys.stderr)
                print('{}: {}'.format(e.__class__.__name__, e), file=sys.stderr)
                sys.exit(1)
    else:
        raise ValueError("unknown input format '{}'".format(format))


class PIGEdge(NamedTuple):
    s: int
    t: int
    tin: int  # 1 for present, -1 for not present, 0 for unknown
    description: Optional[str]


class PIG(NamedTuple):
    vnames: Sequence[str]
    edges: Sequence[PIGEdge]
    cc: Optional[Sequence[int]]
    cclist: Optional[Sequence[Sequence[int]]]


def create_graph(rows, crows, cond):
    # infer vertices or prune edges
    vnames_set = set()
    for row in itertools.chain(rows, crows):
        vnames_set.add(row['from'])
        vnames_set.add(row['to'])
    vnames = list(vnames_set)
    vname_to_index = {vname: i for (i, vname) in enumerate(vnames)}

    cond = cond or {}
    old_rows = rows
    rows = []
    for row in old_rows:
        valid = True
        for k, v in cond.items():
            if k in row and row[k] != v:
                valid = False
        for k, v in row.items():
            if k not in STDKEYS and (k not in cond or cond[k] != v):
                valid = False
        if valid:
            rows.append(row)
    old_crows = crows
    crows = []
    for row in old_crows:
        valid = True
        for k, v in cond.items():
            if k not in row or row[k] != v:
                valid = False
        for k, v in row.items():
            if k not in STDKEYS and k in cond and cond[k] != v:
                valid = False
        if valid:
            crows.append(row)

    edges = []
    for row_group, tin in [(rows, 1), (crows, -1)]:
        for row in row_group:
            s = vname_to_index[row['from']]
            t = vname_to_index[row['to']]
            edge = PIGEdge(s=s, t=t, tin=tin, description=row.get('description'))
            edges.append(edge)
    return PIG(vnames=vnames, edges=edges, cc=None, cclist=None)


def get_SCC(G, adj, radj):
    n = len(G.vnames)
    fintime_order = []
    visited = [False] * n

    def visit1(u):
        if not visited[u]:
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    visit1(v)
            fintime_order.append(u)

    for r in range(n):
        if not visited[r]:
            visit1(r)

    visited = [False] * n
    cc = [None] * n
    cclist = []

    def visit2(u, cci):
        if not visited[u]:
            visited[u] = True
            cc[u] = cci
            cclist[-1].append(u)
            for v in radj[u]:
                if not visited[v]:
                    visit2(v, cci)

    cci = 0
    for r in reversed(fintime_order):
        if not visited[r]:
            cclist.append([])
            visit2(r, cci)
            cci += 1

    return (cc, cclist)


def process_graph(G, tins_to_show):
    n = len(G.vnames)

    def tclose(adj, tadj):
        for s in range(n):
            visited = [False] * n

            def visit(u):
                if not visited[u]:
                    visited[u] = True
                    for v in adj[u]:
                        visit(v)
            visit(s)
            for v in range(n):
                if visited[v]:
                    tadj[s].add(v)

    adj = [set() for v in range(n)]
    radj = [set() for v in range(n)]
    tadj = [set() for v in range(n)]
    tradj = [set() for v in range(n)]

    for e in G.edges:
        if e.tin == 1:
            adj[e.s].add(e.t)
            radj[e.t].add(e.s)
    tclose(adj, tadj)
    tclose(radj, tradj)
    cc, cclist = get_SCC(G, adj, radj)

    """
    print('connected components:', file=sys.stderr)
    for U in cclist:
        if len(U) >= 2:
            print([G.vnames[v] for v in U], file=sys.stderr)
    """

    neg_edges = {}
    for e in G.edges:
        if e.tin == -1:
            neg_edges[(e.s, e.t)] = e
            if e.t in tadj[e.s]:
                print('[error] neg edge ({s}, {t}) is invalid'.format(
                    s=G.vnames[e.s], t=G.vnames[e.t]), file=sys.stderr)
                return G
    new_edges = [e for e in G.edges if e.tin == 1]

    for e in G.edges:
        if e.tin == -1:
            for u in tadj[e.s]:
                for v in tradj[e.t]:
                    if u != v and (u, v) not in neg_edges:
                        neg_edges[(u, v)] = PIGEdge(u, v, tin=-1,
                            description='{s} !-> {t}, {s} -> {u}, {v} -> {t}'.format(
                                s=G.vnames[e.s], t=G.vnames[e.t], u=G.vnames[u], v=G.vnames[v]))
    for u in range(n):
        for v in range(n):
            if u == v or v in tadj[u]:
                continue
            elif (u, v) in neg_edges:
                new_edges.append(neg_edges[(u, v)])
            else:
                new_edges.append(PIGEdge(u, v, tin=0, description=None))

    final_edges = [e for e in new_edges if e.tin in tins_to_show]
    return PIG(G.vnames, final_edges, cc=cc, cclist=cclist)


def get_dot_line(u, v, options):
    if options:
        options_s = ' [' + ','.join(['{}={}'.format(k, v) for k, v in options.items()]) + ']'
    else:
        options_s = ''
    return 'v{} -> v{}{};'.format(u, v, options_s)


def get_dot(G):
    lines = ['digraph G {']
    for i, U in enumerate(G.cclist):
        if len(U) >= 2:
            lines.append('subgraph cluster' + str(i) + ' {')
            lines.append('color=black;')
            for v in U:
                lines.append('v{} [label="{}"];'.format(v, G.vnames[v]))
            lines.append('}')
        else:
            for v in U:
                lines.append('v{} [label="{}"];'.format(v, G.vnames[v]))
    for e in G.edges:
        options = {}
        if e.tin == 0:
            options['style'] = 'dashed'
            options['constraint'] = 'false'
        elif e.tin == -1:
            options['color'] = 'red'
            options['constraint'] = 'false'
        # if e.description:
            # options['label'] = json.dumps(e.description)
        lines.append(get_dot_line(e.s, e.t, options))
    lines.append('}\n')
    return lines


def present_output(G, ofpath, format):
    if format is None:
        format = infer_format(ofpath)
    if format == 'csv':
        with open(ofpath, 'w', newline='') as fp:
            fieldnames = ['from', 'to', 'tin', 'description']
            writer = csv.writer(fp)
            writer.writerow(fieldnames)
            for edge in G.edges:
                row = [G.vnames[edge.s], G.vnames[edge.t], edge.tin, edge.description]
                writer.writerow(row)
    elif format in ('dot', 'svg', 'png', 'pdf'):
        lines = get_dot(G)
        if format == 'dot':
            with open(ofpath, 'w') as fp:
                fp.write('\n'.join(lines))
        else:
            try:
                cp = subprocess.run(['dot', '-T' + format, '-o', ofpath],
                    text=True, input='\n'.join(lines))
                cp.check_returncode()
            except FileNotFoundError as e:
                if e.filename == 'dot':
                    print("Command 'dot' not found. Please install Graphviz.", file=sys.stderr)
                    sys.exit(1)
    else:
        raise NotImplementedError("format '{}' is not implemented yet.".format(format))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # input
    parser.add_argument('fpath', help='path to CSV/JSON file containing conditional implications')
    parser.add_argument('--iformat', choices=['json', 'csv'], help='input format')
    parser.add_argument('--ce', help='path to CSV/JSON file containing counterexamples')
    parser.add_argument('--ceformat', choices=['json', 'csv'], help='counterexample format')
    # output
    parser.add_argument('-o', '--output', required=True, help='output file path')
    parser.add_argument('-f', '--format', choices=['csv', 'dot', 'svg', 'png', 'pdf'],
        help='output format (default: infer from --output)')
    # processing
    parser.add_argument('--raw', action='store_true', default=False,
        help='output raw graph (without processing)')
    parser.add_argument('--tins', help='comma-separated list of tins to show')
    parser.add_argument('--cond',
        help='comma-and-equals-separated dict of conditions to impose')
    args = parser.parse_args()

    rows = read_file(args.fpath, args.iformat)
    if args.ce is not None:
        crows = read_file(args.ce, args.ceformat)
    else:
        crows = []
    G = create_graph(rows, crows, cesep_to_dict(args.cond))

    if args.tins is None:
        if args.ce is None:
            tins_to_show = [1]
        elif args.raw:
            tins_to_show = [-1, 1]
        else:
            tins_to_show = [0, 1]
    else:
        tins_to_show = [int(x) for x in args.tins.strip().split(',')]
    if not args.raw:
        G = process_graph(G, tins_to_show)
    present_output(G, args.output, args.format)


if __name__ == '__main__':
    main()
