import io
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from attendance_tool_v7_1.core.pipeline import match_region, recognize_regions
from attendance_tool_v7_1.core.ocr_engine import RegionReader


class PrefixTests(unittest.TestCase):
    def test_only_configured_leading_prefix_is_ignored(self):
        self.assertEqual(match_region(['业火圣歌小婵'], ['小婵'], ['业火圣歌'])[0], '小婵')
        self.assertIsNone(match_region(['业火圣歌小婵'], ['小婵'])[0])
        self.assertIsNone(match_region(['别的小婵'], ['小婵'], ['业火圣歌'])[0])
        self.assertIsNone(match_region(['小婵业火圣歌'], ['小婵'], ['业火圣歌'])[0])
        self.assertEqual(match_region(['【业火】业火圣歌·小婵'], ['小婵'], ['业火圣歌'])[0], '小婵')
        self.assertEqual(match_region(['小婵'], ['业火圣歌小婵'], ['业火圣歌'])[0], '业火圣歌小婵')

    def test_collisions_and_empty_name_remain_manual(self):
        self.assertIsNone(match_region(['业火圣歌小婵'], ['小婵', '业火圣歌小婵'], ['业火圣歌'])[0])
        self.assertIsNone(match_region(['业火圣歌'], ['业火圣歌'], ['业火圣歌'])[0])

    def test_yy_only_and_confidence_and_evidence(self):
        image = np.zeros((10, 12, 3), np.uint8)
        for source, score, expected in [('YY', .99, {'小婵'}), ('游戏', .99, set()), ('YY', .50, set())]:
            result = recognize_regions(image, source, ['小婵'], manual=((0,0,100,100),1,1),
                reader=lambda crop: [{'text': '业火圣歌小婵', 'score': score}], ignored_prefixes=['业火圣歌'])
            self.assertEqual(result[0], expected)
            self.assertEqual(result[3][0]['texts'], ['业火圣歌小婵'])

    def test_multiple_prefixes_choose_longest_once(self):
        self.assertEqual(match_region(['业火圣歌小婵'], ['小婵'], ['业火', '业火圣歌'])[0], '小婵')
        self.assertEqual(match_region(['圣歌小婵'], ['小婵'], ['业火圣歌', '圣歌'])[0], '小婵')

    def test_name_reader_avoids_full_ocr_when_prefix_matches(self):
        crop = np.zeros((10,12,3), np.uint8)
        reader = RegionReader('YY', ['小婵'])
        reader.ignored_prefixes = ['业火圣歌']
        reader.cache[id(crop)] = [{'text': '业火圣歌小婵', 'score': .99}]
        with patch('attendance_tool_v7_1.core.ocr_engine.load_ocr') as full_ocr:
            self.assertEqual(reader(crop)[0]['text'], '业火圣歌小婵')
            full_ocr.assert_not_called()

    def test_shared_roster_app_and_setting_invalidates_cache(self):
        def file(name, data):
            f = io.BytesIO(data)
            f.name = name
            return f
        _, image = cv2.imencode('.png', np.zeros((10,12,3), np.uint8))
        files = {'names_game': file('game.txt', '小婵\n'.encode()), 'names_yy': None,
                 'game': [], 'yy': [file('yy.png', image.tobytes())]}
        class Reader:
            def __init__(self, *args): pass
            def __call__(self, crop): return [{'text':'业火圣歌小婵', 'score': .99}]
        with patch('streamlit.file_uploader', side_effect=lambda *a, **k: files[k['key']]), \
             patch('attendance_tool_v7_1.core.ocr_engine.RegionReader', Reader), \
             patch('attendance_tool_v7_1.core.pipeline.detect_regions', return_value=[dict(box=(0,0,12,10), reliable=True, method='test')]):
            at = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
            at.checkbox(key='shared_roster').check().run(timeout=30)
            at.text_area(key='ignored_yy_prefix_text').set_value('业火圣歌').run(timeout=30)
            next(b for b in at.button if '开始识别' in b.label).click().run(timeout=30)
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state['yy_found'], {'小婵'})
            self.assertEqual(at.session_state['yy_regions'][0]['texts'], ['业火圣歌小婵'])
            at.text_area(key='ignored_yy_prefix_text').set_value('').run(timeout=30)
            self.assertFalse(at.exception)
            self.assertNotIn('recognized', at.session_state)
            self.assertNotIn('ocr_cache_yy', at.session_state)


if __name__ == '__main__': unittest.main()
