import streamlit as st
from ultralytics import YOLO
from PIL import Image
import collections
import numpy as np

# --- 0. 介面與 CSS 樣式設定 ---
st.set_page_config(page_title="AI 麻將計算平台", layout="centered")

st.markdown("""
    <style>
        .stButton > button {
            border: 2px solid #333 !important; background-color: white !important;
            height: 100px !important; width: 80px !important; margin: 2px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            border-radius: 8px !important;
        }
        .stButton > button div p { font-size: 70px !important; color: #1B1B3A !important; font-family: "Segoe UI Emoji" !important; }
        
        /* 算台模式樣式 */
        .win-tile-box { background-color: #FFF9E6; padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 20px; }
        .section-header { font-size: 24px; font-weight: bold; color: #1B1B3A; margin: 20px 0 10px 0; border-bottom: 3px solid #CCCCFF; padding-bottom: 5px; }
        .count-badge { background-color: #1B1B3A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 18px; margin-left: 10px; vertical-align: middle; }
        .result-label { font-size: 20px; font-weight: bold; margin-bottom: 5px; opacity: 0.8; }
        .wind-info { background-color: #e2e6ea; padding: 10px; border-radius: 8px; font-size: 14px; color: #555; margin-bottom: 15px; text-align: center;}
        .tai-number { font-size: 3.5rem; font-weight: 800; line-height: 1.2; font-family: sans-serif; margin-right: 5px; }
        .tai-text { font-size: 3.5rem; font-weight: 800; line-height: 1.2; font-family: sans-serif; }
        .swap-btn-container { text-align: center; margin: 10px 0; }
        .swap-btn-container button { height: 40px !important; width: 250px !important; font-size: 18px !important; background-color: #f0f2f6 !important; border: 1px solid #ccc !important; }
        
        /* 聽牌模式樣式 */
        .stButton > button[kind="primary"] { height: auto !important; width: 100% !important; padding: 10px !important; background-color: #f0f2f6 !important; border: 1px solid #ccc !important; border-radius: 8px !important; }
        .stButton > button[kind="primary"] div p { font-size: 22px !important; color: #31333F !important; font-family: sans-serif !important; }
        .result-box { padding: 30px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .result-title { font-size: 32px; font-weight: bold; margin-bottom: 20px; opacity: 0.9; }
        .result-content { font-size: 40px; font-weight: 800; line-height: 1.4; }
        .waiting-tiles-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-top: 20px; }
        .waiting-tile { background-color: #fff; border: 2px solid #333; border-radius: 8px; padding: 10px 20px; font-size: 60px; line-height: 1; display: flex; flex-direction: column; align-items: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
        .waiting-name { font-size: 20px; font-weight: normal; margin-top: 5px; color: #555; }
        .error-msg { font-size: 28px; font-weight: bold; color: #721c24; }
        .hint-msg { font-size: 20px; color: #666; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🀄️ AI麻將計算平台")

# --- 1. 核心設定區 ---
st.sidebar.title("⚙️ 核心設定")
app_mode = st.sidebar.radio("📌 功能選擇", ["台數計算", "聽牌分析"])
model_choice = st.sidebar.selectbox("辨識模型", ("yolov8s(2).pt", "yolov8n(2).pt", "YOLOv8s_obb.pt", "YOLOv8n_obb.pt"))

if app_mode == "台數計算":
    flower_mode = st.sidebar.radio("花牌玩法", ["莊家花 (莊家為東)", "開門花 (骰子開門處為東)"])
    dice_val = st.sidebar.number_input("骰子點數", min_value=3, max_value=18, value=7) if flower_mode == "開門花 (骰子開門處為東)" else 0
    st.sidebar.info("本工具用於胡牌計算台數，拍攝時須包含手牌以及門前牌區域。")
else:
    st.sidebar.info("本工具進行聽牌分析，請確保手牌符合3n+1 張的聽牌規範。")
    flower_mode, dice_val = None, 0

@st.cache_resource
def load_yolo_model(name): return YOLO(name)
model = load_yolo_model(model_choice)

TILE_INFO = {
    '1w': {'name': '一萬', 'icon': '🀇', 'w': 1, 'type': 'w', 'val': 1}, '2w': {'name': '二萬', 'icon': '🀈', 'w': 2, 'type': 'w', 'val': 2},
    '3w': {'name': '三萬', 'icon': '🀉', 'w': 3, 'type': 'w', 'val': 3}, '4w': {'name': '四萬', 'icon': '🀊', 'w': 4, 'type': 'w', 'val': 4},
    '5w': {'name': '五萬', 'icon': '🀋', 'w': 5, 'type': 'w', 'val': 5}, '6w': {'name': '六萬', 'icon': '🀌', 'w': 6, 'type': 'w', 'val': 6},
    '7w': {'name': '七萬', 'icon': '🀍', 'w': 7, 'type': 'w', 'val': 7}, '8w': {'name': '八萬', 'icon': '🀎', 'w': 8, 'type': 'w', 'val': 8},
    '9w': {'name': '九萬', 'icon': '🀏', 'w': 9, 'type': 'w', 'val': 9},
    '1D': {'name': '一筒', 'icon': '🀙', 'w': 11, 'type': 'D', 'val': 1}, '2D': {'name': '二筒', 'icon': '🀚', 'w': 12, 'type': 'D', 'val': 2},
    '3D': {'name': '三筒', 'icon': '🀛', 'w': 13, 'type': 'D', 'val': 3}, '4D': {'name': '四筒', 'icon': '🀜', 'w': 14, 'type': 'D', 'val': 4},
    '5D': {'name': '五筒', 'icon': '🀝', 'w': 15, 'type': 'D', 'val': 5}, '6D': {'name': '六筒', 'icon': '🀞', 'w': 16, 'type': 'D', 'val': 6},
    '7D': {'name': '七筒', 'icon': '🀟', 'w': 17, 'type': 'D', 'val': 7}, '8D': {'name': '八筒', 'icon': '🀠', 'w': 18, 'type': 'D', 'val': 8},
    '9D': {'name': '九筒', 'icon': '🀡', 'w': 19, 'type': 'D', 'val': 9},
    '1s': {'name': '一條', 'icon': '🀐', 'w': 21, 'type': 's', 'val': 1}, '2s': {'name': '二條', 'icon': '🀑', 'w': 22, 'type': 's', 'val': 2},
    '3s': {'name': '三條', 'icon': '🀒', 'w': 23, 'type': 's', 'val': 3}, '4s': {'name': '四條', 'icon': '🀓', 'w': 24, 'type': 's', 'val': 4},
    '5s': {'name': '五條', 'icon': '🀔', 'w': 25, 'type': 's', 'val': 5}, '6s': {'name': '六條', 'icon': '🀕', 'w': 26, 'type': 's', 'val': 6},
    '7s': {'name': '七條', 'icon': '🀖', 'w': 27, 'type': 's', 'val': 7}, '8s': {'name': '八條', 'icon': '🀗', 'w': 28, 'type': 's', 'val': 8},
    '9s': {'name': '九條', 'icon': '🀘', 'w': 29, 'type': 's', 'val': 9},
    'ew': {'name': '東', 'icon': '🀀', 'w': 31, 'type': 'z'}, 'sw': {'name': '南', 'icon': '🀁', 'w': 32, 'type': 'z'},
    'ww': {'name': '西', 'icon': '🀂', 'w': 33, 'type': 'z'}, 'nw': {'name': '北', 'icon': '🀃', 'w': 34, 'type': 'z'},
    'zhong': {'name': '中', 'icon': '🀄︎', 'w': 35, 'type': 'z'}, 'fa': {'name': '發', 'icon': '🀅', 'w': 36, 'type': 'z'},
    'wd': {'name': '白', 'icon': '🀆', 'w': 37, 'type': 'z'},
    '1rf': {'name': '春', 'icon': '🀦', 'w': 51, 'type': 'h', 'suit': 'rf', 'v': 1}, '2rf': {'name': '夏', 'icon': '🀧', 'w': 52, 'type': 'h', 'suit': 'rf', 'v': 2},
    '3rf': {'name': '秋', 'icon': '🀨', 'w': 53, 'type': 'h', 'suit': 'rf', 'v': 3}, '4rf': {'name': '冬', 'icon': '🀩', 'w': 54, 'type': 'h', 'suit': 'rf', 'v': 4},
    '1bf': {'name': '梅', 'icon': '🀢', 'w': 55, 'type': 'h', 'suit': 'bf', 'v': 1}, '2bf': {'name': '蘭', 'icon': '🀣', 'w': 56, 'type': 'h', 'suit': 'bf', 'v': 2},
    '3bf': {'name': '竹', 'icon': '🀤', 'w': 57, 'type': 'h', 'suit': 'bf', 'v': 3}, '4bf': {'name': '菊', 'icon': '🀥', 'w': 58, 'type': 'h', 'suit': 'bf', 'v': 4}
}

# --- 2. 演算法邏輯 ---

# A. 算台模式用：支援槓牌 (4張) 的主拆解引擎
def recursive_decompose_main(counts, sets_needed, win_tile, current_sets=[]):
    if sum(counts.values()) == 0: return (sets_needed == 0), current_sets
    if sets_needed <= 0: return False, []

    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    for take in [4, 3]:
        if counts[tile] >= take:
            temp = counts.copy(); temp[tile] -= take
            ok, res = recursive_decompose_main(temp, sets_needed - 1, win_tile, current_sets + [(f'set_{take}', tile)])
            if ok: return True, res

    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info.get('val', 0) <= 7:
        t2 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+1), None)
        t3 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+2), None)
        if t2 and t3 and counts.get(t2,0) > 0 and counts.get(t3,0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[t2]-=1; temp[t3]-=1
            seq = [tile, t2, t3]; pos = seq.index(win_tile) if win_tile in seq else -1
            ok, res = recursive_decompose_main(temp, sets_needed - 1, win_tile, current_sets + [('seq', seq, pos)])
            if ok: return True, res
    return False, []

# B. 聽牌分析用：嚴格遵守 3 張一組的面子拆解
def recursive_decompose_waiting(counts, sets_needed):
    if sum(counts.values()) == 0: return (sets_needed == 0)
    if sets_needed <= 0: return False
    
    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    if counts[tile] >= 3:
        temp = counts.copy(); temp[tile] -= 3
        if recursive_decompose_waiting(temp, sets_needed - 1): return True
        
    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info.get('val', 0) <= 7:
        t2 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+1), None)
        t3 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+2), None)
        if t2 and t3 and counts.get(t2,0) > 0 and counts.get(t3,0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[t2]-=1; temp[t3]-=1
            if recursive_decompose_waiting(temp, sets_needed - 1): return True
    return False

def check_hu_for_waiting(counts):
    for eye in counts:
        if counts[eye] >= 2:
            temp = counts.copy(); temp[eye] -= 2
            rem_tiles = sum(temp.values())
            if rem_tiles % 3 != 0: continue
            sets_needed = rem_tiles // 3
            if recursive_decompose_waiting(temp, sets_needed): return True
    return False

def get_waiting_tiles(hand_codes):
    counts = collections.Counter(hand_codes)
    waiting = []
    all_tiles = [k for k,v in TILE_INFO.items() if v['type'] != 'h']
    for t in all_tiles:
        temp = counts.copy()
        temp[t] += 1
        if temp[t] > 4: continue 
        if check_hu_for_waiting(temp):
            waiting.append(t)
    return waiting

# --- 3. 聽牌模式防呆與判定 ---
def analyze_waiting_status(con):
    hand_only = [c for c in con if TILE_INFO[c]['type'] != 'h']
    total_counts = collections.Counter(con)

    for code, count in total_counts.items():
        info = TILE_INFO[code]
        if info['type'] != 'h' and count > 4:
            return "error", f"牌數錯誤：**{info['name']}** 有 {count} 張 (單一牌種上限為 4)", []

    hand_len = len(hand_only)
    if hand_len > 16:
        return "error", f"手牌數量為 {hand_len} 張。<br>手牌上限限制為 n=5 (最多 16 張)。", []
    if hand_len % 3 == 0:
        return "error", f"手牌數量為 {hand_len} 張 (相公)。<br>若要聽牌，手牌應為 3n+1 張。", []
    elif hand_len % 3 == 2:
        return "error", f"手牌數量為 {hand_len} 張 (3n+2)。<br>這是已經胡牌或未打牌的數量，請移除一張多餘的牌以計算聽牌。", []

    waiting_list = get_waiting_tiles(hand_only)
    if waiting_list: return "waiting", "聽牌中！", waiting_list
    else: return "not_waiting", "尚未聽牌", []

# --- 4. 算台模式邏輯 ---
def run_full_logic(con, exp, win_tile, streak, dealer_p, is_zm, win_on_dealer, f_mode, dice, manual_list, base_tai, wind_circle):
    all_codes = con + exp
    hand_only = [c for c in all_codes if TILE_INFO[c]['type'] != 'h']
    hua_codes = [c for c in all_codes if TILE_INFO[c]['type'] == 'h']

    total_counts = collections.Counter(all_codes)
    for code, count in total_counts.items():
        info = TILE_INFO[code]
        limit = 1 if info['type'] == 'h' else 4
        if count > limit: return False, "相公", [f"偵測到 **{info['name']}** 有 {count} 張 (上限 {limit})"], None
    if any(TILE_INFO[c]['type'] == 'h' for c in con):
        return False, "相公", ["手牌區不可含花牌"], None

    all_counts = collections.Counter(hand_only)
    hu_ok, best_sets, win_is_eye = False, [], False
    for eye, count in all_counts.items():
        if count >= 2:
            temp = all_counts.copy(); temp[eye] -= 2
            ok, res = recursive_decompose_main(temp, 5, win_tile)
            if ok: hu_ok = True; best_sets = res; win_is_eye = (eye == win_tile); break
    if not hu_ok: return False, "相公", ["結構錯誤 (無法湊成5面子+1眼)"], None

    hand_minus_win = list([c for c in con if TILE_INFO[c]['type'] != 'h'])
    if win_tile in hand_minus_win: hand_minus_win.remove(win_tile)
    waiting_list = get_waiting_tiles(hand_minus_win)
    is_strict_single_wait = (len(waiting_list) == 1 and waiting_list[0] == win_tile)

    tai, details = 0, []
    n_counts = collections.Counter([TILE_INFO[c]['name'] for c in hand_only])
    suits = set([TILE_INFO[c]['type'] for c in hand_only])

    if all(t == 'z' for t in suits): tai += 16; details.append("字一色 16台")
    elif len(suits - {'z'}) == 1:
        if 'z' in suits: tai += 4; details.append("混一色 4台")
        else: tai += 8; details.append("清一色 8台")
        
    d_tri = sum(1 for d in ['中','發','白'] if n_counts[d] >= 3)
    d_pair = sum(1 for d in ['中','發','白'] if n_counts[d] == 2)
    if d_tri == 3: tai += 8; details.append("大三元 8台")
    elif d_tri == 2 and d_pair == 1: tai += 4; details.append("小三元 4台")
    else:
        if n_counts['中'] >= 3: tai += 1; details.append("紅中 1台")
        if n_counts['發'] >= 3: tai += 1; details.append("發財 1台")
        if n_counts['白'] >= 3: tai += 1; details.append("白板 1台")
        
    w_tri = sum(1 for w in ['東','南','西','北'] if n_counts[w] >= 3)
    is_big_four = (w_tri == 4)
    if is_big_four: tai += 16; details.append("大四喜 16台 (不加計圈風與字牌門風)")
    elif w_tri == 3 and any(n_counts[w] == 2 for w in ['東','南','西','北']): tai += 8; details.append("小四喜 8台")
        
    if all(s[0].startswith('set') for s in best_sets): tai += 4; details.append("碰碰胡 4台")
    
    con_hand = [c for c in con if TILE_INFO[c]['type'] != 'h']
    is_quan_qiu = (len(con_hand) == 2 and win_is_eye)

    wait_type = None
    if not is_quan_qiu:
        if is_strict_single_wait and win_is_eye: wait_type = "單吊 1台"
        elif is_strict_single_wait:
            for s in best_sets:
                if s[0] == 'seq' and s[2] != -1:
                    v = TILE_INFO[win_tile]['val']
                    if s[2] == 1: wait_type = "中洞 1台"
                    elif (s[2] == 0 and v == 7) or (s[2] == 2 and v == 3): wait_type = "邊張 1台"
                    else: wait_type = "單吊 1台"
    if wait_type: tai += 1; details.append(wait_type)

    anke_count = 0
    exposed_counts = collections.Counter(exp)
    for s in best_sets:
        ctype = s[0]
        if ctype in ['set_3', 'set_4']:
            tile = s[1]
            if exposed_counts[tile] >= 3: exposed_counts[tile] -= 3
            else: anke_count += 1

    if anke_count >= 3:
        tm={3:2, 4:5, 5:8}; tai += tm.get(anke_count, 0); details.append(f"{anke_count}暗刻 {tm.get(anke_count,0)}台")
        
    is_menqing = (len(exp) == 0)
    if is_quan_qiu: tai += 2; details.append("全求人 2台 (含單吊)")
    elif is_menqing:
        if is_zm and "槓上開花" not in manual_list and "海底撈月" not in manual_list: tai += 3; details.append("門清一摸三 3台")
        elif not is_zm: tai += 1; details.append("門清 1台")
        elif (is_zm and ("槓上開花" in manual_list or "海底撈月" in manual_list)): tai += 1; details.append("門清 1台")
            
    if len(hua_codes)==0 and not any(TILE_INFO[c]['type']=='z' for c in hand_only) and all(s[0]=='seq' for s in best_sets) and not is_menqing and not wait_type:
        tai += 2; details.append("平胡 2台")
        
    dealer_map_idx = {"我": 0, "下家(右)": 1, "對家(對面)": 2, "上家(左)": 3}
    dealer_idx_rel = dealer_map_idx[dealer_p]

    if f_mode.startswith("莊家"):
        logical_east_idx_rel = dealer_idx_rel
        calc_note = "莊家位置"
    else:
        dice_offset = (dice - 1) % 4
        logical_east_idx_rel = (dealer_idx_rel + dice_offset) % 4
        calc_note = f"骰子{dice}點開門位置"
        
    my_wind_idx = (4 - logical_east_idx_rel) % 4
    wind_names = ["東", "南", "西", "北"]
    my_wind_name = wind_names[my_wind_idx]
    my_flower_num = my_wind_idx + 1
    wind_debug_info = f"判斷基準：{calc_note} <br> 我的門風：<b>{my_wind_name}風</b> (對應花牌：{my_flower_num}花)"
    
    if not is_big_four:
        if n_counts[wind_circle] >= 3: tai += 1; details.append(f"圈風({wind_circle}風) 1台")
        if n_counts[my_wind_name] >= 3: tai += 1; details.append(f"門風({my_wind_name}風) 1台")
            
    if len(hua_codes) == 8: tai += 8; details.append("八仙過海 8台")
    elif len(hua_codes) == 7: tai += 7; details.append("七搶一 7台")
    else:
        h_suits = collections.Counter([TILE_INFO[c]['suit'] for c in hua_codes])
        gang_suits = [s for s, c in h_suits.items() if c == 4]
        for g in gang_suits: tai += 2; details.append(f"花槓 ({'春夏秋冬' if g=='rf' else '梅蘭竹菊'}) 2台")
        loose_flowers = [c for c in hua_codes if TILE_INFO[c]['suit'] not in gang_suits]
        for c in loose_flowers:
            if TILE_INFO[c]['v'] == my_flower_num: tai += 1; details.append(f"方位花牌({TILE_INFO[c]['name']}) 1台")
                
    tai += base_tai; details.insert(0, f"底台 {base_tai}台")

    has_streak = False
    if dealer_p == "我": has_streak = True
    elif is_zm: has_streak = True
    elif win_on_dealer: has_streak = True; tai += 1; details.append("胡莊家 1台")

    if has_streak and streak > 0: tai += (2*streak); details.append(f"連{streak}拉{streak} {2*streak}台")
    if dealer_p == "我": tai += 1; details.append("莊家 1台")
        
    if "槓上開花" in manual_list: tai += 2; details.append("槓上自摸 2台")
    elif "海底撈月" in manual_list: tai += 2; details.append("海底自摸 2台")
    elif is_zm and not is_menqing: tai += 1; details.append("自摸 1台")
        
    manual_score_map = {"天胡": 16, "搶槓": 1, "河底撈魚": 1, "咪幾": 8, "哩咕": 8, "天地人胡": 16}
    for m in manual_list:
        if m in ["槓上開花", "海底撈月"]: continue
        pts = manual_score_map.get(m, 0)
        if m == "天胡": pts = 16
        if m in ["搶槓", "河底撈魚", "咪幾", "哩咕", "天地人胡"]:
            real_pts = pts if pts > 0 else 1
            if m == "地人胡": real_pts = 16
            if m in ["咪幾", "哩咕"]: real_pts = 8
            tai += real_pts; details.append(f"{m} {real_pts}台")
            
    return True, tai, details, wind_debug_info

# --- 5. 影像偵測處理 (徹底修復存取與座標計算) ---
def process_detection(image_obj, source_type, current_model_name, mode):
    img_id_base = getattr(image_obj, 'name', 'camera') if source_type == 'upload' else 'camera_shot'
    cache_key = (img_id_base, current_model_name, mode) 

    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        st.session_state.current_cache_key = cache_key
        st.session_state.current_image = image_obj

        results = model(image_obj)
        st.session_state.current_plot = results[0].plot()

        tile_data = []
        for r in results:
            # 🌟 修正：正確提取 Box 座標，避免變數錯誤導致資料消失
            if hasattr(r, 'obb') and r.obb is not None:
                classes = r.obb.cls.cpu().numpy()
                xywh = r.obb.xywhr.cpu().numpy()
                for i, c in enumerate(classes):
                    tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})
            elif hasattr(r, 'boxes') and r.boxes is not None:
                classes = r.boxes.cls.cpu().numpy()
                xywh = r.boxes.xywh.cpu().numpy()
                for i, c in enumerate(classes):
                    tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})
                    
        if not tile_data:
            st.warning("未偵測到任何麻將牌")
            st.session_state.con_manual, st.session_state.exp_manual = [], []
            return

        if mode == "台數計算":
            # 🌟 算台模式：計算 Y 軸斷層，將牌區分為上下兩排
            sorted_y = sorted(tile_data, key=lambda x: x['y'])
            gaps = np.diff([d['y'] for d in sorted_y])
            max_idx = np.argmax(gaps) if len(gaps) > 0 else -1
            # 確保斷層大於 40 pixel 才算兩排，否則全歸為手牌
            threshold = (sorted_y[max_idx]['y'] + sorted_y[max_idx+1]['y'])/2 if (max_idx != -1 and gaps[max_idx] > 40) else -1
            
            st.session_state.con_manual = [d['code'] for d in tile_data if d['y'] >= threshold]
            st.session_state.exp_manual = [d['code'] for d in tile_data if d['y'] < threshold]

            # 抓取手牌最右側作為胡牌張
            hand_objs = [d for d in tile_data if d['y'] >= threshold]
            if hand_objs: st.session_state.win_tile = max(hand_objs, key=lambda d: d['x'])['code']
            elif tile_data: st.session_state.win_tile = tile_data[0]['code']
        else:
            # 🌟 聽牌模式：不分區域，通通直接視為手牌
            st.session_state.con_manual = [d['code'] for d in tile_data]
            st.session_state.exp_manual = []

# --- 6. UI 渲染主程式 ---
def render_main_ui(mode):
    if 'current_image' not in st.session_state:
        st.info(" ☝️  請先上傳照片或使用相機拍照，AI 將自動辨識手牌。")
        return
        
    st.image(st.session_state.current_plot, caption=f"AI 辨識結果 ({model_choice})", use_container_width=True)
    all_codes = st.session_state.con_manual + st.session_state.exp_manual
    st.markdown(f'<div class="section-header"> 🎴  牌面管理 <span class="count-badge">偵測總數：{len(all_codes)} 張</span></div>', unsafe_allow_html=True)

    # 🔹 算台模式介面 🔹
    if mode == "台數計算":
        with st.container():
            st.markdown('<div class="win-tile-box">', unsafe_allow_html=True)
            if 'win_tile' not in st.session_state: st.session_state.win_tile = all_codes[0] if all_codes else '1w'
            win_info = TILE_INFO.get(st.session_state.win_tile, {'icon':'?', 'name':'未知'})
            
            st.write(f"#### 目前胡牌張：{win_info['name']}")
            col1, col2 = st.columns([1, 1])
            with col1: st.button(win_info['icon'], key="win_now", use_container_width=True)
            with col2:
                with st.popover(" 🔄  更改胡牌張"):
                    st.write("選擇新的胡牌張：")
                    all_keys = sorted([i for i in TILE_INFO.items() if i[1]['type'] != 'h'], key=lambda x: x[1]['w'])
                    grid = st.columns(4)
                    counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                    for idx, (k, v) in enumerate(all_keys):
                        with grid[idx % 4]:
                            if st.button(v['icon'], key=f"sw_{k}", disabled=(counts_all[k] >= 4)): 
                                st.session_state.win_tile = k; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.write(f"🐹 手牌：")
        codes = st.session_state.con_manual
        s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
        cols = st.columns(11)
        for i, idx in enumerate(s_idx):
            with cols[i % 11]:
                if st.button(TILE_INFO[codes[idx]]['icon'], key=f"h_{i}"): st.session_state.con_manual.pop(idx); st.rerun()
        with st.popover(f" ➕  新增手牌"):
            p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
            counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
            for k, v in all_keys:
                limit = 1 if v['type'] == 'h' else 4
                if st.button(v['icon'], key=f"add_h_{k}", disabled=(counts_all[k] >= limit)): st.session_state.con_manual.append(k); st.rerun()
                
        st.markdown('<div class="swap-btn-container">', unsafe_allow_html=True)
        if st.button(" 🔃  交換手牌與門前牌", help="點擊互換上下兩區的牌"):
            st.session_state.con_manual, st.session_state.exp_manual = st.session_state.exp_manual, st.session_state.con_manual
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write(f"🐥 門前牌：")
        codes = st.session_state.exp_manual
        s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
        cols = st.columns(11)
        for i, idx in enumerate(s_idx):
            with cols[i % 11]:
                if st.button(TILE_INFO[codes[idx]]['icon'], key=f"d_{i}"): st.session_state.exp_manual.pop(idx); st.rerun()
        with st.popover(f" ➕  新增門前"):
            p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
            counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
            for k, v in all_keys:
                limit = 1 if v['type'] == 'h' else 4
                if st.button(v['icon'], key=f"add_d_{k}", disabled=(counts_all[k] >= limit)): st.session_state.exp_manual.append(k); st.rerun()
                
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            base_t = st.number_input("底台數", min_value=0, value=3)
            dealer = st.selectbox("誰是莊家", ["我", "下家(右)", "對家(對面)", "上家(左)"])
            wind_r = st.selectbox("目前風圈", ["東", "南", "西", "北"])
        with c2:
            m_list = st.multiselect("手動加台：", ["搶槓", "海底撈月", "河底撈魚", "槓上開花", "咪幾", "哩咕", "天地人胡"])
            force_zm = "海底撈月" in m_list or "槓上開花" in m_list
            if force_zm:
                st.info(" 💡  已選擇槓上/海底，系統自動設為自摸")
                is_zm = True; win_on_dealer = False
            else:
                is_zm = st.checkbox("我是自摸", value=False)
                win_on_dealer = st.checkbox("胡莊家 (莊家放槍)") if dealer != "我" and not is_zm else False
                
        streak = st.number_input("連莊次數", min_value=0, value=0) if dealer == "我" or win_on_dealer else 0
        
        hu_ok, res_tai, details, wind_info = run_full_logic(st.session_state.con_manual, st.session_state.exp_manual, st.session_state.win_tile, streak, dealer, is_zm, win_on_dealer, flower_mode, dice_val, m_list, base_t, wind_r)
        
        if res_tai == "相公":
            html_content = f'''<div style="background-color:#f8d7da; color:#721c24; padding:20px; border-radius:12px; text-align:center;"><div class="result-label"> 🏆️ 預估台數</div><div style="margin-top: 10px;"><span class="tai-number">相公 👻 </span></div></div>'''
        else:
            html_content = f'''<div style="background-color:#d4edda; color:#155724; padding:20px; border-radius:12px; text-align:center;"><div class="result-label"> 🏆️ 預估台數</div><div style="display: flex; justify-content: center; align-items: baseline; margin-top: 10px;"><span class="tai-number">{res_tai}</span><span class="tai-text">台</span></div></div>'''
            
        st.markdown(html_content, unsafe_allow_html=True)
        col_re_1, col_re_2 = st.columns([4, 1])
        with col_re_2:
            if st.button(" 🔄  重新計算", key="refresh_btn", help="依照目前設定重新計算台數"): st.rerun()
        if wind_info: st.markdown(f'<div class="wind-info">{wind_info}</div>', unsafe_allow_html=True)
        for d in details: st.write(f" 📌  {d}")

    # 🔹 聽牌模式介面 🔹
    else:
        st.write(f"🐹 手牌：")
        codes = st.session_state.con_manual
        s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
        cols = st.columns(11)
        for i, idx in enumerate(s_idx):
            with cols[i % 11]:
                if st.button(TILE_INFO[codes[idx]]['icon'], key=f"h_{i}"):
                    st.session_state.con_manual.pop(idx); st.rerun()

        with st.popover(f" ➕  新增手牌"):
            st.write("點擊圖示加入：")
            all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
            cols_add = st.columns(8)
            counts = collections.Counter(st.session_state.con_manual)
            for idx, (k, v) in enumerate(all_keys):
                with cols_add[idx % 8]:
                    limit = 1 if v['type'] == 'h' else 4
                    if st.button(v['icon'], key=f"add_h_{k}", disabled=(counts[k] >= limit)):
                        st.session_state.con_manual.append(k); st.rerun()
                        
        st.markdown("---")
        status, title, data = analyze_waiting_status(st.session_state.con_manual)

        if status == "waiting":
            bg_color, text_color = "#cce5ff", "#004085"
            icon_html = "".join([f'<div class="waiting-tile"><div>{TILE_INFO[t]["icon"]}</div><div class="waiting-name">{TILE_INFO[t]["name"]}</div></div>' for t in data])
            html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color};"><div class="result-title">👀 聽牌分析</div><div class="result-content">🔥 {title}</div><div style="margin-top: 10px; font-size: 20px;">這手牌聽以下這些牌：</div><div class="waiting-tiles-container">{icon_html}</div></div>"""
        elif status == "not_waiting":
            bg_color, text_color = "#fff3cd", "#856404"
            html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color};"><div class="result-title">👀 聽牌分析</div><div class="result-content">{title} 🤡</div><div class="hint-msg">目前還沒有聽牌呦!。</div></div>"""
        else: 
            bg_color, text_color = "#f8d7da", "#721c24"
            html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color};"><div class="result-title">⚠️ 牌型異常</div><div class="error-msg">相公 👻</div><div class="hint-msg">{title}</div></div>"""
            
        st.markdown(html_content, unsafe_allow_html=True)
        st.write("")
        if st.button(" 🔄  重新分析", key="refresh_all", use_container_width=True, type="primary"): st.rerun()

# --- 啟動入口 ---
t1, t2 = st.tabs([" 📷︎ ︎即時拍照", " 📁 上傳照片"])
with t1:
    cam = st.camera_input("拍照")
    if cam: process_detection(Image.open(cam), 'camera', model_choice, app_mode)
with t2:
    up = st.file_uploader("選照片", type=['png', 'jpg', 'jpeg'])
    if up: process_detection(Image.open(up), 'upload', model_choice, app_mode)

# 如果「相機沒拍」且「上傳區沒檔案」，但「快取還有舊圖」，就清除快取讓 UI 回歸初始狀態
if not cam and not up:
    keys_to_clear = ['current_image', 'current_plot', 'con_manual', 'exp_manual', 'current_cache_key']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

render_main_ui(app_mode)




















