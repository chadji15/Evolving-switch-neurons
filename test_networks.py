import unittest
import solve
import eval

class TestNetworks(unittest.TestCase):

    def test_one_to_one(self):
        agent = solve.solve_one_to_one_3x3()
        optimal_score = 940
        scores = [eval.eval_one_to_one_3x3(agent) for _ in range(100)]
        for score in scores:
            self.assertGreaterEqual(score, optimal_score)

    # def test_one_to_many(self):
    #     agent = solve.solve_one_to_many()
    #     optimal_score = 1964
    #     scores = [eval.eval_one_to_many_3x2(agent) for _ in range(100)]
    #     for score in scores:
    #         self.assertGreaterEqual(score,optimal_score)

    def test_t_maze(self):
        agent = solve.solve_tmaze()
        optimal_score = 28
        evaluator = eval.TmazeEvaluator()
        scores = [evaluator.eval_tmaze(agent) for _ in range(100)]
        for score in scores:
            self.assertGreaterEqual(score, optimal_score)

if __name__ == '__main__':
    unittest.main()