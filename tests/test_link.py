import pytest
from dlk.utils.parser import BaseConfigParser, LinkUnionTool


@pytest.mark.general
class TestLinkUnionTool(object):
    """Assisting tool for parsering the "_link" of config. All the function named the top level has high priority than low level

    This class is mostly for resolve the confilicts of the low and high level register links.
    """

    @pytest.mark.parametrize(("top_links", 'low_links', "result"), [ 
                                 [ # simple only top link case
                                     {
                                         "para_1": "para_2"
                                     },
                                     {},
                                     {
                                         "para_1": ["para_2"]
                                     }
                                 ],
                                 [ # simple only low-link case
                                     {},
                                     {
                                         "para_1": "para_3"
                                     },
                                     {
                                         "para_1": ["para_3"]
                                     }
                                 ],
                                 [ # when a 'link-from' is appeared many times, all the 'link-to' will be unioned and be the new 'link_to'
                                     {
                                         "para_1": "para_2"
                                     },
                                     {
                                         "para_1": "para_3"
                                     },
                                     {
                                         "para_1": ["para_2", "para_3"]
                                     }
                                 ],
                                 [ # when top-link and low-link all link a key to the same 'link-to'
                                     {
                                         "para_1": "para_2"
                                     },
                                     {
                                         "para_3": "para_2"
                                     },
                                     {
                                         "para_1": ["para_2", "para_3"]
                                     }
                                 ],
                                 [ # assert para_2 is repeated assignment
                                     {
                                         "para_1": "para_2",
                                         "para_3": "para_2"
                                     },
                                     {
                                     },
                                     AssertionError
                                 ],
                                 [ # assert para_2 is repeated assignment
                                     {
                                         "para_1": "para_2",
                                         "para_1": "para_2"
                                     },
                                     {
                                     },
                                     {
                                         "para_1": ["para_2"]
                                     }
                                 ],
                                 [ # low-link will be reversed when the `link-to` is appeared in the top-level link
                                     {
                                         "para_1": "para_2",
                                     },
                                     {
                                         "para_1": "para_3",
                                         "para_4": "para_2", # will be reversed as "para_2": "para_4"
                                     },
                                     {
                                         'para_1': ['para_2', 'para_3', 'para_4']
                                     }
                                 ],
                                 [ # The `link_from` and `link_to` has been linked to different values, but now you want to link them together.
                                     {
                                     },
                                     {
                                         "para_3": "para_1",
                                         "para_4": "para_2",
                                         "para_1": "para_2",
                                     },
                                     PermissionError
                                 ],
                                 [ # WARNING: This config will not raise an error. But use the rule of low-link will be reversed when the `link-to` is appeared in the top-level link
                                     {
                                         "para_1": "para_2",
                                     },
                                     {
                                         "para_3": "para_1",
                                         "para_4": "para_2",
                                     },
                                     {
                                         "para_1": ["para_2", "para_3", "para_4"]
                                     }
                                 ],
                             ])
    def test_link_union(self, top_links, low_links, result):
        """TODO: Docstring for test_link_union.
        """
        if not isinstance(result, dict):
            try:
                link_union = LinkUnionTool()
                link_union.register_top_links(top_links)
                link_union.register_low_links(low_links)
                raise PermissionError("No covered error: result is ", link_union.get_links())
            except Exception as e:
                assert type(e) == result
        else:
            link_union = LinkUnionTool()
            link_union.register_top_links(top_links)
            link_union.register_low_links(low_links)
            assert link_union.get_links() == result
