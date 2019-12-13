import unittest
import index

class IndexTestCase(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(index.sigmoid(0), .5, "should be equal")
        self.assertGreater(index.sigmoid(100), .9, "should be greater than .9")
        self.assertLess(index.sigmoid(-100), .9, "should be less than .9")

    def test_siglayer_output(self):
        sl = index.SigmoidLayer(3)
        self.assertEqual([1,.5,0], sl.output_values([100,0,-800]))

    def test_siglayer_backprop(self):
        sl = index.SigmoidLayer(3)
        # output 0.8807970779778823   0.11920292202211755 1.0
        ipt, error = [2, -2, 40], [-0.8807970779778823, 1-0.11920292202211755, -1.0]
        sl.output_values(ipt)
        [self.assertAlmostEqual(r,e) for (r, e) in zip (sl.backprop(error), [-0.092478043, 0.092478043, 0])]

    def test_linlayer_output(self):
        pass