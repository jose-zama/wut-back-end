import unittest
from ModelWrapper import predict


class ModelTest(unittest.TestCase):
    def test_two_transactions(self):
        # set
        transactions = 'uber\nstarbucks'

        # execute
        result = predict(transactions)

        # assert
        lines = result.split('\n')
        headers = lines[0]
        self.assertEqual('Details,Prediction', headers)

        self.assertRegex(lines[1], '^uber,.*$')

        self.assertRegex(lines[2], '^starbucks,.*$')


if __name__ == '__main__':
    unittest.main()
