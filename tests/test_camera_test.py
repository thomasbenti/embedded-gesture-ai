import unittest
from types import SimpleNamespace

from embedded_gesture_ai.camera_test import recognize_gesture


def make_hand(extended: set[int], thumb_extended: bool = False):
    landmarks = [SimpleNamespace(x=0.5, y=0.7) for _ in range(21)]
    for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
        landmarks[pip].y = 0.5
        landmarks[tip].y = 0.3 if tip in extended else 0.7

    landmarks[3].x = 0.5
    landmarks[4].x = 0.3 if thumb_extended else 0.5
    return landmarks


class RecognizeGestureTests(unittest.TestCase):
    def test_recognizes_point(self):
        hand = make_hand({8})
        self.assertEqual(recognize_gesture(hand, "Right"), "Point")

    def test_recognizes_peace_sign(self):
        hand = make_hand({8, 12})
        self.assertEqual(recognize_gesture(hand, "Right"), "Peace")

    def test_recognizes_open_palm(self):
        hand = make_hand({8, 12, 16, 20})
        self.assertEqual(recognize_gesture(hand, "Right"), "Open palm")

    def test_recognizes_fist_and_thumbs_up(self):
        self.assertEqual(recognize_gesture(make_hand(set()), "Right"), "Fist")
        self.assertEqual(
            recognize_gesture(make_hand(set(), thumb_extended=True), "Right"),
            "Thumbs up",
        )

    def test_unclassified_pose_and_invalid_landmarks(self):
        self.assertEqual(recognize_gesture(make_hand({8, 16}), "Right"), "Hand")
        self.assertEqual(recognize_gesture([], "Right"), "Hand")


if __name__ == "__main__":
    unittest.main()
