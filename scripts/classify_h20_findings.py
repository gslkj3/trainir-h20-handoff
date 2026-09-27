#!/usr/bin/env python3
"""Read-only triage: prints locations and expression types, never source values.

Does not approve publication, modify source, or bypass the collector.
"""
import argparse
import ast
import json
from pathlib import Path
import re


URL = re.compile(r'https?://([^\s/:@]+):([^\s/@]+)@')
VARIABLE = re.compile(r'(?:\$[A-Za-z_][A-Za-z_0-9]*|\$\{[A-Za-z_][A-Za-z_0-9]*\}|\$\{\{\s*[^{}\r\n]+\s*\}\})')


def classify(text, line, rule, suffix):
    if rule == 'literal_secret' and suffix == '.py':
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return {'classification': 'python_parse_unavailable_manual_review'}
        matches = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if getattr(node, 'lineno', None) == line:
                    value = node.value
                    kind = type(value).__name__
                    if isinstance(value, ast.Constant):
                        kind = 'Constant:' + type(value.value).__name__
                    matches.append(kind)
        if not matches:
            return {'classification': 'unresolved_python_line_manual_review'}
        review = any(x in ('Constant:str', 'Constant:bytes', 'JoinedStr', 'BinOp') for x in matches)
        return {'classification': ('possible_literal_or_constructed_value_review' if review
                                   else 'not_a_direct_literal_assignment'),
                'expression_types': sorted(set(matches))}
    if rule == 'credential_in_url':
        lines = text.splitlines()
        if line < 1 or line > len(lines):
            return {'classification': 'line_not_found'}
        matches = list(URL.finditer(lines[line - 1]))
        if not matches:
            return {'classification': 'pattern_no_longer_matches'}
        categories = []
        for match in matches:
            categories.append('password_is_variable_reference' if VARIABLE.fullmatch(match.group(2))
                              else 'password_contains_literal_or_unresolved_expression')
        return {'classification': 'url_userinfo_review', 'password_categories': categories}
    return {'classification': 'manual_review_required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--megatron', required=True)
    parser.add_argument('--galvatron', required=True)
    args = parser.parse_args()
    roots = {'Megatron-LM': Path(args.megatron).resolve(),
             'Hetu-Galvatron-dtsir': Path(args.galvatron).resolve()}
    report = json.loads(Path(args.report).read_text(encoding='utf-8-sig'))
    output = []
    for finding in report.get('suspected_secrets', []):
        row = {k: finding[k] for k in ('path', 'line', 'rule')}
        try:
            label, relative = finding['path'].split('/', 1)
            root = roots[label]
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
            text = path.read_text(encoding='utf-8-sig')
            row.update(classify(text, int(finding['line']), finding['rule'], path.suffix))
        except (OSError, ValueError, KeyError, UnicodeError):
            row['classification'] = 'cannot_safely_read_source'
        output.append(row)
    print(json.dumps({'scope': 'Triage only; no source values emitted; no export approval.',
                      'findings': output}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
