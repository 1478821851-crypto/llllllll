import io
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from attendance_tool_v7_1.core.batch import recognize_images


class BatchTests(unittest.TestCase):
    def test_merge_deduplicates_members_and_preserves_independent_evidence(self):
        images = [dict(image=np.full((10, 20, 3), i, np.uint8), name=f'{i}.png', key=str(i)) for i in range(2)]
        calls = []
        def recognizer(image, source, names, status, manual):
            calls.append((int(image[0, 0, 0]), manual))
            region = dict(id=1, box=(0, 0, 20, 10), reliable=True, crop=image.copy(), matched='甲', error=None)
            fallback = dict(id=2, box=(0, 0, 20, 10), reliable=False, crop=image.copy(), matched=None, error='故障证据')
            return {'甲'}, [{'区域': 2, '标准姓名': '乙'}], ['甲'], [region, fallback]
        result = recognize_images(images, '游戏', ['甲', '乙'], manual_settings={'1': '独立网格'}, recognizer=recognizer)
        found, uncertain, texts, regions = result
        self.assertEqual(found, {'甲'})
        self.assertEqual(texts, ['甲'])
        self.assertEqual([r['id'] for r in regions], [1, 2, 3, 4])
        self.assertEqual([r['区域'] for r in uncertain], [2, 4])
        self.assertEqual([r['image_name'] for r in regions], ['0.png', '0.png', '1.png', '1.png'])
        self.assertEqual(int(regions[2]['crop'][0, 0, 0]), 1)
        self.assertEqual(regions[3]['error'], '故障证据')
        self.assertEqual(calls, [(0, None), (1, '独立网格')])

    def test_empty_source(self):
        self.assertEqual(recognize_images([], 'YY', []), (set(), [], [], []))

    def test_app_two_game_two_yy_and_removal_invalidates_results(self):
        def upload(name, data):
            f = io.BytesIO(data)
            f.name = name
            return f
        def image_file(name, intensity):
            ok, data = cv2.imencode('.png', np.full((12, 24, 3), intensity, np.uint8))
            assert ok
            return upload(name, data.tobytes())
        files = {'names_game': upload('游戏名单.txt', '甲\n乙\n'.encode()),
                 'names_yy': upload('YY名单.txt', '语音甲\n语音乙\n'.encode()),
                 'game': [image_file('游戏1.png', 20), image_file('游戏2.png', 40)],
                 'yy': [image_file('YY1.png', 60), image_file('YY2.png', 80)]}
        def uploader(*args, **kwargs):
            if kwargs['key'] in ('game', 'yy'):
                self.assertTrue(kwargs['accept_multiple_files'])
            return files[kwargs['key']]
        calls = []
        def recognizer(image, source, names, status, manual):
            calls.append((source, int(image[0, 0, 0])))
            name = '甲' if image[0, 0, 0] in (20, 60) else '乙'
            self.assertEqual(names, ['甲', '乙'] if source == '游戏' else ['语音甲', '语音乙'])
            if source == 'YY':
                name = '语音' + name
            return {name}, [], [name], [dict(id=1, box=(0,0,24,12), reliable=True,
                crop=image.copy(), matched=name, error=None, texts=[name], suggestions=[], method='测试')]
        app = Path(__file__).parents[1] / 'app.py'
        with patch('streamlit.file_uploader', side_effect=uploader), patch('attendance_tool_v7_1.core.batch.recognize_regions', side_effect=recognizer):
            at = AppTest.from_file(str(app)).run(timeout=30)
            self.assertFalse(at.exception)
            at.button[4].click().run(timeout=30)
            self.assertFalse(at.exception)
            self.assertEqual(calls, [('游戏',20), ('游戏',40), ('YY',60), ('YY',80)])
            self.assertEqual(at.session_state['game_found'], {'甲','乙'})
            self.assertEqual(at.session_state['yy_found'], {'甲','乙'})
            self.assertEqual([r['id'] for r in at.session_state['game_regions']], [1,2])
            # Changing only the preview must not discard recognized results.
            at.selectbox[0].set_value(1).run(timeout=30)
            self.assertFalse(at.exception)
            self.assertTrue(at.session_state['recognized'])
            files['names_yy'] = upload('YY名单.txt', '语音乙\n语音甲\n'.encode())
            at.run(timeout=30)
            self.assertFalse(at.exception)
            self.assertNotIn('recognized', at.session_state)
            self.assertNotIn('ocr_cache_yy', at.session_state)
            files['game'].pop()
            at.run(timeout=30)
            self.assertFalse(at.exception)
            self.assertNotIn('recognized', at.session_state)


if __name__ == '__main__':
    unittest.main()
