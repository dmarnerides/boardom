import boardom as bd


class TestRecurseGetElements:
    def test_gets_element_single(self):
        i = 0
        for x in bd.recurse_get_elements(1):
            assert x == 1
            i += 1
        assert i == 1

    def test_gets_elements_tuple(self):
        a = (1, 2, 3, 4)
        for i, x in enumerate(bd.recurse_get_elements(a)):
            assert x == a[i]

    def test_gets_elements_list(self):
        a = [1, 2, 3, 4]
        for i, x in enumerate(bd.recurse_get_elements(a)):
            assert x == a[i]

    def test_gets_elements_dict(self):
        a = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        values = [1, 2, 3, 4]
        for i, x in enumerate(bd.recurse_get_elements(a)):
            assert x == values[i]

    def test_gets_elements_nested(self):
        a = [
            0,
            {'a': 1, 'b': 2, 'c': (3, 4, 5), 'd': 6},
            (7, {'e': 8, 'f': 9}),
        ]
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i, x in enumerate(bd.recurse_get_elements(a)):
            assert x == values[i]

    def test_gets_elements_generator(self):
        a = [
            0,
            {'a': 1, 'b': 2, 'c': (3, 4, 5), 'd': 6},
            (7, {'e': 8, 'f': 9}),
        ]
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        gen = bd.recurse_get_elements(a)
        for i, x in enumerate(bd.recurse_get_elements(gen)):
            assert x == values[i]
