from pathlib import Path
import sys
# Streamlit entrypoints run from either repository root or this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import cv2
import threading
import time
import gc
import json

from attendance_tool_v7_1.utils.paired_roster import read_id_list, pair_rosters, remap_result
from attendance_tool_v7_1.utils.image import read_image
from attendance_tool_v7_1.core.batch import recognize_images, load_image, crop_image
from attendance_tool_v7_1.core.adaptive_regions import detect_regions, grid_boxes
import hashlib
from attendance_tool_v7_1.core.export import make_excel


# UI state is separate from the unchanged recognition results.
def reset_review():
    for key in list(st.session_state):
        if key.startswith(("review_", "attendance_editor")):
            del st.session_state[key]
    st.session_state["review_regions"] = {}
    st.session_state["review_overrides"] = {}
    st.session_state["review_revision"] = 0


def effective_presence(prefix, name):
    automatic = name in st.session_state.get(prefix + "_found", set())
    manual = any(source == prefix and person == name
                 for (source, _), person in st.session_state.get("review_regions", {}).items())
    return st.session_state.get("review_overrides", {}).get((prefix, name), automatic or manual)


def confirm_region(prefix, region_id, person, checked_key):
    confirmations = st.session_state.setdefault("review_regions", {})
    old = confirmations.pop((prefix, region_id), None)
    if st.session_state[checked_key]:
        confirmations[(prefix, region_id)] = person
    overrides = st.session_state.setdefault("review_overrides", {})
    for name in (old, person):
        overrides.pop((prefix, name), None)
    st.session_state["review_revision"] = st.session_state.get("review_revision", 0) + 1


def change_region_person(prefix, region_id, checked_key):
    old = st.session_state.setdefault("review_regions", {}).pop((prefix, region_id), None)
    st.session_state.setdefault("review_overrides", {}).pop((prefix, old), None)
    st.session_state[checked_key] = False
    st.session_state["review_revision"] = st.session_state.get("review_revision", 0) + 1


def edit_attendance(editor_key, names):
    # Capture only the user's edits; never replace the OCR baseline.
    overrides = st.session_state.setdefault("review_overrides", {})
    for index, changes in st.session_state[editor_key].get("edited_rows", {}).items():
        name = names[int(index)]
        for column, prefix in (("游戏", "game"), ("YY", "yy")):
            if column in changes:
                overrides[(prefix, name)] = bool(changes[column])


@st.cache_resource
def get_ocr_lock():
    return threading.Lock()

ocr_lock = get_ocr_lock()
st.set_page_config(page_title="人员出勤识别工具 V7.2 Web", page_icon="📋", layout="wide")
st.title("📋 人员出勤识别工具 V7.2 Web")
st.caption("V7.2 Web 先切图再OCR｜自适应游戏网格、YY成员行与通知区分离｜模糊候选仅供人工确认")

# =========================================================
# 上传标准名录
# =========================================================

st.subheader("① 分别上传游戏 ID 与 YY ID 名单")
st.caption("两份名单均为单列，按相同顺序排列：第1行对应同一个人，第2行对应同一个人。支持 TXT、CSV、XLSX；请勿留空行或重复 ID。")
roster_columns = st.columns(2)
with roster_columns[0]:
    game_name_file = st.file_uploader("游戏 ID 名单", type=["xlsx", "csv", "txt"], key="names_game")
with roster_columns[1]:
    yy_name_file = st.file_uploader("YY ID 名单", type=["xlsx", "csv", "txt"], key="names_yy")
source_names = {"game": [], "yy": []}
roster_error = None
paired_members = []
try:
    for prefix, upload, source in [("game", game_name_file, "游戏"), ("yy", yy_name_file, "YY")]:
        if upload:
            source_names[prefix] = read_id_list(upload, source)
            st.caption(f"{source}名单：{len(source_names[prefix])} 行")
    if game_name_file and yy_name_file:
        paired_members = pair_rosters(source_names['game'], source_names['yy'])
except (ValueError, UnicodeError, OSError) as exc:
    roster_error = str(exc)
    st.error(roster_error)
except Exception:
    roster_error = "名单读取失败，请检查文件格式；TXT、CSV 请使用 UTF-8 编码。"
    st.error(roster_error)
standard_names = [member['游戏ID'] for member in paired_members]
yy_by_game = {member['游戏ID']: member['YYID'] for member in paired_members}
source_aliases = {
    "game": {member['游戏ID']: member['游戏ID'] for member in paired_members},
    "yy": {member['YYID']: member['游戏ID'] for member in paired_members},
}
def member_label(game_id):
    return f"游戏：{game_id} ｜ YY：{yy_by_game[game_id]}" if game_id is not None else "请选择人员"
if paired_members:
    st.success(f"已按行对应 {len(paired_members)} 位成员。请检查对应关系后开始识别。")
    with st.expander("查看游戏 ID 与 YY ID 对应关系"):
        st.dataframe(pd.DataFrame(paired_members), hide_index=True, width="stretch")
elif not roster_error:
    st.info("请上传两份名单；如果两边 ID 相同，可以在两侧上传同一个文件。")

# =========================================================
# 图片
# =========================================================

st.divider()

st.subheader("② 上传截图（游戏、YY均支持多张）")
st.caption("同一侧可以继续拖入图片追加，也可以一次选择多张；点击文件旁的 × 可移除。识别时合并全部截图，同一成员不会重复计数。")
col1, col2 = st.columns(2)
with col1:
    game_files = st.file_uploader("🎮 游戏截图（可多张）", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True, key="game")
with col2:
    yy_files = st.file_uploader("🎧 YY截图（可多张）", type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True, key="yy")

# Length delimiters preserve file boundaries and distinguish game from YY.
input_digest = hashlib.sha256()
for prefix, uploads in [("names_game", [game_name_file] if game_name_file else []),
                        ("names_yy", [yy_name_file] if yy_name_file else []),
                        ("game", game_files), ("yy", yy_files)]:
    input_digest.update(prefix.encode())
    for uploaded in uploads:
        data = uploaded.getvalue()
        input_digest.update(len(data).to_bytes(8, "big"))
        input_digest.update(data)
        input_digest.update(uploaded.name.encode())
fingerprint = input_digest.hexdigest()
if st.session_state.get("input_fingerprint") != fingerprint:
    for key in list(st.session_state):
        if key.startswith(("game_", "yy_", "ocr_cache_")) or key in ("recognized", "attendance_editor"):
            del st.session_state[key]
    reset_review()
    st.session_state["input_fingerprint"] = fingerprint

def decode_uploads(uploads, source):
    images = []
    for index, uploaded in enumerate(uploads):
        image = read_image(uploaded)
        if image is None:
            st.error(f"{source}图片无法读取：{uploaded.name}，请移除后重新上传。")
            continue
        images.append(dict(data=uploaded.getvalue(), name=uploaded.name, key=f"{index}_{hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]}"))
    return images

game_images = decode_uploads(game_files, "游戏")
yy_images = decode_uploads(yy_files, "YY")
game_image = bool(game_images) or None
yy_image = bool(yy_images) or None
for column, source, prefix, images in [(col1, "游戏", "game", game_images), (col2, "YY", "yy", yy_images)]:
    with column:
        if images:
            st.success(f"已上传 {len(images)} 张{source}截图")
            selected = st.selectbox(f"查看{source}图片", range(len(images)),
                format_func=lambda i, items=images: f"第{i+1}张 · {items[i]['name']}", key=prefix+"_selected_image")
            preview = load_image(images[selected])
            if preview.shape[1] > 1200:
                scale = 1200 / preview.shape[1]
                preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption=images[selected]['name'], width="stretch")
            del preview

manual_settings = {"game": {}, "yy": {}}
with st.expander("✂️ 切图预览 / 调整（每张截图单独设置）"):
    st.caption("每张截图分别自动切图，不拼接原图。特殊界面可为对应图片单独设置手动网格，调整后需重新开始识别。")
    for source, prefix, images in [("游戏", "game", game_images), ("YY", "yy", yy_images)]:
        for index, item in enumerate(images, 1):
            stem = prefix + "_" + item['key']
            label = f"{source}第{index}张 · {item['name']}"
            st.markdown(f"**{label}**")
            manual = st.checkbox(f"{label}使用手动切图", key=stem+"_manual")
            settings = None
            if manual:
                x = st.slider(f"{label}左右范围（%）", 0, 100, (0,100), key=stem+"_x")
                y = st.slider(f"{label}上下范围（%）", 0, 100, (0,100), key=stem+"_y")
                rows = st.number_input(f"{label}行数", 1, 300, 1, key=stem+"_rows")
                columns = st.number_input(f"{label}列数", 1, 30, 1, key=stem+"_cols")
                settings = ((x[0], y[0], x[1], y[1]), int(rows), int(columns))
            manual_settings[prefix][item['key']] = settings
            if st.button(f"预览{label}切图", key=stem+"_preview"):
                image = load_image(item)
                regions = grid_boxes(image, *settings) if manual else detect_regions(image, source)
                overlay = image.copy()
                for region in regions:
                    if region['reliable']:
                        x1,y1,x2,y2 = region['box']
                        cv2.rectangle(overlay, (x1,y1), (x2-1,y2-1), (0,200,0), max(1,image.shape[1]//700))
                st.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB), caption=f"{label}：{sum(r['reliable'] for r in regions)} 个切图区域", width="stretch")
                del image, overlay, regions

# =========================================================
# 开始识别
# =========================================================

st.divider()

if st.button(
    "🚀 开始识别并生成出勤表",
    type="primary",
    width="stretch"
):

    if not standard_names:

        st.error(
            "请先上传两份有效且行数一致的 ID 名单。"
        )

        st.stop()

    if len(game_images) != len(game_files) or len(yy_images) != len(yy_files):
        st.error("有图片无法读取，请先移除或重新上传，避免漏掉这张图的成员。")
        st.stop()

    if (
        game_image is None
        and yy_image is None
    ):

        st.error(
            "至少上传一张可读取的截图。"
        )

        st.stop()
    
        # 全局 OCR 锁：同一时间只允许一个用户进入识别
    if not ocr_lock.acquire(blocking=False):
        st.warning("⏳ 当前有其他用户正在识别，请稍后再试。")
        st.stop()

    started = time.perf_counter()
    progress = st.progress(0, text="任务开始：切图、加载模型、识别、生成结果")
    try:
        for source_index, (source, prefix, images) in enumerate([("游戏", "game", game_images), ("YY", "yy", yy_images)]):
            found, uncertain, texts, regions = set(), [], [], []
            if images:
                status = st.status(f"{source}：正在切图 / 首次加载模型可能较慢", expanded=True)
                digest = hashlib.sha256()
                for item in images:
                    digest.update(len(item['data']).to_bytes(8, 'big'))
                    digest.update(item['data'])
                    digest.update(item['name'].encode())
                digest.update(json.dumps([source, paired_members, manual_settings[prefix]], ensure_ascii=False).encode())
                cache_key = digest.hexdigest()
                # One result per source per session. Never cache exceptions or
                # partial OCR results; review state is separate from this cache.
                cached = st.session_state.get("ocr_cache_" + prefix)
                if cached and cached[0] == cache_key:
                    found, uncertain, texts, regions = cached[1]
                    status.write("相同整批截图、名录和切图设置：复用上次 OCR 结果，无需重新运算。")
                else:
                    found, uncertain, texts, regions = recognize_images(
                        images, source, source_names[prefix], status, manual_settings[prefix], compress_crops=True)
                    found, uncertain, texts, regions = remap_result((found, uncertain, texts, regions), source_aliases[prefix])
                    if regions and any(r['reliable'] for r in regions) and not any(r['error'] for r in regions):
                        st.session_state["ocr_cache_" + prefix] = (cache_key, (found, uncertain, texts, regions))
                    else:
                        st.session_state.pop("ocr_cache_" + prefix, None)
                count = sum(r['reliable'] for r in regions)
                errors = sum(bool(r['error']) for r in regions)
                if errors:
                    status.update(label=f"{source}：{errors} 个区域异常，请人工核对或重试", state="error", expanded=True)
                    status.warning(f"{source}：切出 {count} 个区域，确认 {len(found)} 人；{errors} 个区域OCR异常，原裁图已保留。")
                elif not count:
                    status.update(label=f"{source}：未切出区域，请调整切图", state="complete", expanded=True)
                    status.warning(f"{source}：未成功切出卡片/成员行，原图已保留。请查看切图预览或手动调整。")
                else:
                    status.update(label=f"{source}：识别完成，确认 {len(found)} 人", state="complete", expanded=False)
                    status.success(f"{source}：切出 {count} 个区域，确认 {len(found)} 人；未匹配区域全部保留。")
            st.session_state[prefix+"_found"] = found
            st.session_state[prefix+"_uncertain"] = uncertain
            st.session_state[prefix+"_texts"] = texts
            st.session_state[prefix+"_regions"] = regions
            progress.progress((source_index + 1) / 2, text=f"已完成 {source}，累计 {time.perf_counter() - started:.1f} 秒")
        st.session_state["recognized"] = True
        reset_review()
        st.success(f"任务完成，用时 {time.perf_counter() - started:.1f} 秒。可在下方人工核对和导出。")
    finally:
        try:
            gc.collect()  # Once per task, after result extraction.
        finally:
            ocr_lock.release()


# =========================================================
# 最终结果 + 人工修正
# =========================================================

if (
    st.session_state.get(
        "recognized",
        False
    )
    and standard_names
):

    game_found = st.session_state.get(
        "game_found",
        set()
    )

    yy_found = st.session_state.get(
        "yy_found",
        set()
    )

    with st.expander("✂️ 查看全部裁图（包括已匹配，便于检查边界）"):
        st.caption("这里可检查自动切图是否完整。人工图库只排除这张裁图自己唯一匹配的成员，不按其他区域的签到结果删图。")
        show_all = st.checkbox("显示全部裁图", key="show_all_crops")
        if show_all:
            for source, prefix in [("游戏", "game"), ("YY", "yy")]:
                st.write(source)
                regions = [r for r in st.session_state.get(prefix+"_regions", []) if r["reliable"]]
                for base in range(0, len(regions), 4):
                    cols = st.columns(4)
                    for col, region in zip(cols, regions[base:base+4]):
                        with col:
                            st.image(cv2.cvtColor(crop_image(region),cv2.COLOR_BGR2RGB), width="stretch")
                            st.caption(f"{region['image_name']} · #{region['id']} · {member_label(region['matched']) if region['matched'] else '待人工核对'}")

    with st.expander("🖼️ 人工核对图库：所有未匹配裁图及原图兜底", expanded=True):
        st.caption("空白OCR、识别异常、姓名歧义都保留原裁图。原图兜底用于发现自动切图遗漏。候选不会自动签到，可直接在每张图下确认；取消后同步撤销该区域的确认。原图兜底每次只确认一人，其余遗漏可在最终表补充。")
        tabs = st.tabs(["🎮 游戏", "🎧 YY"])
        for tab, prefix in zip(tabs, ["game", "yy"]):
            with tab:
                unmatched = [r for r in st.session_state.get(prefix+"_regions", []) if not r["matched"]]
                st.write(f"待核对：{len(unmatched)} 个区域（含原图兜底，如有）")
                for base in range(0, len(unmatched), 3):
                    cols = st.columns(3)
                    for col, region in zip(cols, unmatched[base:base+3]):
                        with col:
                            st.image(cv2.cvtColor(crop_image(region), cv2.COLOR_BGR2RGB), width="stretch")
                            st.caption(f"第{region['image_index']}张 · {region['image_name']} · #{region['id']} · {region['method']} · {region['box']}")
                            st.write("OCR：", " / ".join(region["texts"]) or "无可识别文字")
                            if region["error"]:
                                st.warning(region["error"])
                            if region["suggestions"]:
                                st.write("人工候选：", "、".join(member_label(r["标准姓名"]) for r in region["suggestions"]))
                            candidates = list(dict.fromkeys(
                                item["标准姓名"] for item in region["suggestions"]
                                if item["标准姓名"] in standard_names))
                            stem = f"review_{prefix}_{region['id']}"
                            checked_key = stem + "_confirmed"
                            if len(candidates) == 1:
                                person = candidates[0]
                            else:
                                options = candidates + [n for n in standard_names if n not in candidates]
                                person = st.selectbox(
                                    "选择正确人员（候选优先，可搜索完整名录）", [None] + options,
                                    format_func=member_label,
                                    key=stem + "_person", on_change=change_region_person,
                                    args=(prefix, region["id"], checked_key))
                            st.checkbox(
                                f"确认为：{member_label(person)}" if person else "选择人员后确认",
                                key=checked_key, disabled=person is None,
                                on_change=confirm_region,
                                args=(prefix, region["id"], person, checked_key))
                            confirmed = st.session_state.get("review_regions", {}).get((prefix, region["id"]))
                            if confirmed:
                                if effective_presence(prefix, confirmed):
                                    st.success(f"已确认：{member_label(confirmed)}（{'游戏' if prefix == 'game' else 'YY'}）")
                                else:
                                    st.info(f"区域已确认：{member_label(confirmed)}；最终表已人工取消。取消并重新勾选可恢复。")


    rows = []

    for member in paired_members:
        name = member["游戏ID"]
        rows.append(
            {
                "序号": member["序号"],
                "人员": name,
                "YY ID": member["YYID"],
                "游戏": effective_presence("game", name),
                "YY": effective_presence("yy", name)
            }
        )

    edit_df = pd.DataFrame(
        rows
    )


    st.divider()

    st.header(
        "📊 最终出勤结果"
    )

    st.caption(
        "识别错了可以直接点击游戏/YY栏的复选框人工修正。"
    )


    editor_key = f"attendance_editor_{st.session_state.get('review_revision', 0)}"
    for old_key in list(st.session_state):
        if old_key.startswith("attendance_editor_") and old_key != editor_key:
            del st.session_state[old_key]
    edited_df = st.data_editor(
        edit_df,
        width="stretch",
        hide_index=True,
        disabled=["序号", "人员", "YY ID"],
        column_config={
            "人员":
                st.column_config.TextColumn(
                    "游戏 ID"
                ),

            "游戏":
                st.column_config.CheckboxColumn(
                    "游戏"
                ),

            "YY":
                st.column_config.CheckboxColumn(
                    "YY"
                )
        },
        key=editor_key,
        on_change=edit_attendance,
        args=(editor_key, list(standard_names))
    )


    # =====================================================
    # 自动计算
    # =====================================================

    final_df = edited_df.copy()

    final_df["总次数"] = (
        final_df["游戏"].astype(int)
        +
        final_df["YY"].astype(int)
    )

    final_df["出勤"] = (
        final_df["总次数"] == 2
    )


    # 显示版
    display_df = final_df.rename(columns={"人员": "游戏 ID"}).copy()

    display_df["游戏"] = display_df[
        "游戏"
    ].map({
        True: "✅",
        False: "❌"
    })

    display_df["YY"] = display_df[
        "YY"
    ].map({
        True: "✅",
        False: "❌"
    })

    display_df["出勤"] = display_df[
        "出勤"
    ].map({
        True: "✅",
        False: "❌"
    })

    st.subheader(
        "最终确认表"
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


    # =====================================================
    # 统计
    # =====================================================

    game_count = int(
        final_df["游戏"].sum()
    )

    yy_count = int(
        final_df["YY"].sum()
    )

    attendance_count = int(
        final_df["出勤"].sum()
    )


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "总名录",
        len(standard_names)
    )

    c2.metric(
        "游戏",
        game_count
    )

    c3.metric(
        "YY",
        yy_count
    )

    c4.metric(
        "最终出勤",
        attendance_count
    )


    # =====================================================
    # V5.5 Fixed 准确率测试
    # 重要：必须读取当前最终确认表，而不是旧的自动识别集合。
    # 这样上面已经勾选为 True 的人，不可能同时出现在“漏识别”里。
    # =====================================================

    current_game_found = set(
        final_df.loc[final_df["游戏"].astype(bool), "人员"].astype(str).tolist()
    )
    current_yy_found = set(
        final_df.loc[final_df["YY"].astype(bool), "人员"].astype(str).tolist()
    )

    with st.expander("🧪 准确率测试（可选）"):
        st.caption("按当前最终确认表实时计算；上方勾选/取消后，这里的结果会同步更新。测试选择不会反过来修改考勤表。")

        def _show_accuracy_test(label, predicted_set, key_prefix):
            st.markdown(f"#### {label}")
            truth = st.multiselect(
                f"请选择{label}图片里实际出现的人",
                options=standard_names,
                format_func=member_label,
                default=[],
                key=f"{key_prefix}_truth"
            )

            if truth:
                truth_set = set(truth)
                predicted_set = set(predicted_set)
                correct = truth_set & predicted_set
                missed = truth_set - predicted_set
                wrong = predicted_set - truth_set

                precision = len(correct) / len(predicted_set) if predicted_set else 0
                recall = len(correct) / len(truth_set) if truth_set else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

                a, b, c, d = st.columns(4)
                a.metric("正确识别", len(correct))
                b.metric("漏识别", len(missed))
                c.metric("误识别", len(wrong))
                d.metric("F1综合准确率", f"{f1 * 100:.1f}%")

                st.caption(
                    f"精确率 {precision * 100:.1f}% ｜ 召回率 {recall * 100:.1f}% ｜ "
                    "F1用于综合衡量误识别和漏识别。"
                )

                if missed:
                    st.write("漏识别：", "、".join([n for n in standard_names if n in missed]))
                if wrong:
                    st.write("误识别：", "、".join([n for n in standard_names if n in wrong]))
            else:
                st.info("选择这张图片里实际出现的人后，会自动计算。")

        if game_image is not None:
            _show_accuracy_test("游戏", current_game_found, "game")
        if yy_image is not None:
            _show_accuracy_test("YY", current_yy_found, "yy")


    # =====================================================
    # 疑似
    # =====================================================

    uncertain_rows = []

    for item in st.session_state.get(
        "game_uncertain",
        []
    ):

        row = item.copy()
        row["来源"] = "游戏"

        uncertain_rows.append(
            row
        )

    for item in st.session_state.get(
        "yy_uncertain",
        []
    ):

        row = item.copy()
        row["来源"] = "YY"

        uncertain_rows.append(
            row
        )


    if uncertain_rows:

        st.subheader(
            "⚠️ 建议人工确认"
        )

        st.dataframe(
            pd.DataFrame(uncertain_rows).rename(columns={"标准姓名": "游戏 ID"}),
            width="stretch",
            hide_index=True
        )


    # =====================================================
    # 导出
    # =====================================================

    st.divider()

    excel_data = make_excel(
        display_df
    )

    st.download_button(
        "📥 下载最终出勤 Excel",
        data=excel_data,
        file_name="最终出勤表.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch"
    )


    # =====================================================
    # OCR调试
    # =====================================================

    with st.expander(
        "🔍 查看OCR原始结果"
    ):

        st.markdown(
            "#### 游戏"
        )

        st.write(
            st.session_state.get(
                "game_texts",
                []
            )
        )

        st.markdown(
            "#### YY"
        )

        st.write(
            st.session_state.get(
                "yy_texts",
                []
            )
        )
