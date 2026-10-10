"""Read one ID per row without silently dropping rows or duplicate IDs."""
import csv
import io
from collections import Counter

import pandas as pd


def read_id_list(upload, source):
    suffix = upload.name.lower().rsplit('.', 1)[-1]
    data = upload.getvalue()
    if suffix == 'txt':
        rows = [[value] for value in data.decode('utf-8-sig').splitlines()]
    elif suffix == 'csv':
        rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
    elif suffix == 'xlsx':
        table = pd.read_excel(io.BytesIO(data), header=None, dtype=str, keep_default_na=False)
        rows = table.values.tolist()
    else:
        raise ValueError('名单只支持 TXT、CSV、XLSX。')
    if not rows:
        raise ValueError(f'{source}名单为空。')
    if any(len(row) != 1 for row in rows):
        raise ValueError(f'{source}名单须为单列，每行一个 ID；请移除多余列。')
    values = [str(row[0]).strip() for row in rows]
    # Only explicit headers are removed, never an arbitrary first ID.
    headers = {'游戏id', '游戏 id', 'yyid', 'yy id', 'id', '游戏名单', 'yy名单'}
    if values and values[0].casefold() in headers:
        values.pop(0)
    if not values:
        raise ValueError(f'{source}名单没有 ID。')
    empty = [str(i) for i, value in enumerate(values, 1) if not value]
    if empty:
        raise ValueError(f'{source}名单第 {"、".join(empty[:8])} 行为空，请补齐；不会自动删除空行。')
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f'{source}名单存在重复 ID：{"、".join(duplicates[:5])}，请核对，避免关联到多个人。')
    return values


def pair_rosters(game_ids, yy_ids):
    if len(game_ids) != len(yy_ids):
        raise ValueError(f'两份名单行数不一致：游戏 {len(game_ids)} 行、YY {len(yy_ids)} 行。请补齐后再识别。')
    return [dict(序号=i, 游戏ID=game_id, YYID=yy_id)
            for i, (game_id, yy_id) in enumerate(zip(game_ids, yy_ids), 1)]


def remap_result(result, aliases):
    """Use game ID as a unique row key; retain the OCR-side ID for review."""
    found, uncertain, texts, regions = result
    def suggestion(row):
        copied = dict(row)
        copied['识别ID'] = copied['标准姓名']
        copied['标准姓名'] = aliases[copied['标准姓名']]
        return copied
    remapped = []
    for region in regions:
        copied = dict(region)
        copied['matched_id'] = copied['matched']
        if copied['matched']:
            copied['matched'] = aliases[copied['matched']]
        copied['suggestions'] = [suggestion(row) for row in copied['suggestions']]
        remapped.append(copied)
    return {aliases[name] for name in found}, [suggestion(row) for row in uncertain], texts, remapped
