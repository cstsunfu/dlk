import pytest
import json
from dlk.utils.parser import BaseConfigParser
from dlk.core import embedding_config_register

@embedding_config_register('test_base_embedding_config')
class BaseEmbeddingConfig4TestInherit(object):
    """This is for test for inherit"""
    default_config = {
        "_name": "test_base_embedding_config",
        "config": {
            "parent_para1": 1,
            "parent_para2": 2,
            "parent_para3": {
                "parent_para3_1": 31,
                "parent_para3_2": 32,
            },
        }
    }
    def __init__(self, config):
        super(BaseEmbeddingConfig4TestInherit, self).__init__()


@pytest.mark.general
class TestBaseConfigParser(object):

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                                [
                                    {
                                        "_name": "base_config",
                                        "config": {
                                            "para1": 1,
                                            "para2": 2,
                                            "nest_config": {
                                                "nest_para1": 3,
                                                "nest_para2": 4
                                            }
                                        },
                                    },
                                    [{
                                        "_name": "base_config",
                                        "config": {
                                            "para1": 1,
                                            "para2": 2,
                                            "nest_config": {
                                                "nest_para1": 3,
                                                "nest_para2": 4
                                            }
                                        },
                                    }]
                                ]
                            ])
    def test_basic_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        assert json.dumps(config.parser(), sort_keys=True) == json.dumps(result, sort_keys=True)

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [
                        {
                            "_name": "base_config",
                            "imodel": {
                                "config": {
                                    "para1": 1,
                                    "para2": 2,
                                    "para3": 3,
                                    "para4": 4,
                                    "para5": 5,
                                    "nest": {
                                         "nest_para6": 6,
                                         "nest_para7": 7,
                                         "nest_para8": 8
                                    },
                                    "para9": 9,
                                    "_link": {
                                        "para1": 'para2',
                                        "para3": ['para4', 'para5'],
                                        "nest.nest_para6": ['nest.nest_para7'],
                                        "nest.nest_para8": ['para9'],
                                    }
                                },
                             },
                        },
                        [
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "para1": 1,
                                        "para2": 1,
                                        "para3": 3,
                                        "para4": 3,
                                        "para5": 3,
                                        "nest": {
                                             "nest_para6": 6,
                                             "nest_para7": 6,
                                             "nest_para8": 8
                                        },
                                        "para9": 8,
                                    },
                                 },
                            },
                        ]
                    ],
                    [
                        {
                            "_name": "base_config",
                            "imodel": {
                                "config": {
                                    "para1": 1,
                                    "para2": 2,
                                    "nest": {
                                        "nest_para6": 6,
                                        "nest_para7": 7,
                                        "_link": {
                                            "nest_para6": ['nest_para7'],
                                        }
                                    },
                                    "_link": {
                                        "para1": 'para2',
                                    }
                                },
                             },
                        },
                        [
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "para1": 1,
                                        "para2": 1,
                                        "nest": {
                                             "nest_para6": 6,
                                             "nest_para7": 6,
                                        },
                                    },
                                 },
                            },
                        ]
                    ],
                ])
    def test_link_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [ # basic search, search value is list of values
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "nest_config": {
                                    "nest_para1": 3,
                                    "nest_para2": 4
                                },
                                "_search": {
                                    "para1": [5, 6],
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 2,
                                    "nest_config": {
                                        "nest_para1": 3,
                                        "nest_para2": 4
                                    },
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 2,
                                    "nest_config": {
                                        "nest_para1": 3,
                                        "nest_para2": 4
                                    },
                                },
                            },
                        ]
                    ],

                    [ # basic search, search candidates should be a string which will be evaluate as a list of values
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "nest_config": {
                                    "nest_para1": 3,
                                    "nest_para2": 4
                                },
                                "_search": {
                                    "para1": "list(range(5, 7))",
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 2,
                                    "nest_config": {
                                        "nest_para1": 3,
                                        "nest_para2": 4
                                    },
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 2,
                                    "nest_config": {
                                        "nest_para1": 3,
                                        "nest_para2": 4
                                    },
                                },
                            },
                        ]
                    ],

                    [ # multi search, search result should be cartesian prod
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "_search": {
                                    "para1": "list(range(5, 7))",
                                    "para2": ['a', 'b'],
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'b',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'b',
                                },
                            },
                        ]
                    ],
                ])
    def test_basic_search_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [ # basic search, search candidates should be a string which will be evaluate as a list of values
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "_search": {
                                    "para1": "list(range(5, 7))",
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 2,
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 2,
                                },
                            },
                        ]
                    ],
                    [ # multi search, search result should be cartesian prod
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "_search": {
                                    "para1": "list(range(5, 7))",
                                    "para2": ['a', 'b'],
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'b',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'b',
                                },
                            },
                        ]
                    ],
                ])
    def test_evalable_search_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        # print(json.dumps(config.parser(), sort_keys=True, indent=2))
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [ # multi search, search result should be cartesian prod
                        {
                            "_name": "base_config",
                            "config": {
                                "para1": 1,
                                "para2": 2,
                                "_search": {
                                    "para1": "list(range(5, 7))",
                                    "para2": ['a', 'b'],
                                }
                            },
                        },
                        [
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'a',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 5,
                                    "para2": 'b',
                                },
                            },
                            {
                                "_name": "base_config",
                                "config": {
                                    "para1": 6,
                                    "para2": 'b',
                                },
                            },
                        ]
                    ],
                ])
    def test_multi_search_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        # print(json.dumps(config.parser(), sort_keys=True, indent=2))
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [ # recursive search, recursive search para  must have been registed in parsers
                        {
                            "_name": "base_config",
                            "imodel": {
                                "config": {}, # recursive search must for the para which has registed in parser
                                "model": {},
                                "_search": {
                                    "config": [
                                        {
                                            "nest_para": "*@*",
                                            "_search": {
                                                "nest_para": [5, 6],
                                            },
                                        },
                                        {
                                            "nest_para": 'a',
                                        }
                                    ],
                                     "model": [
                                         {
                                             "config":{"model_para1": 1}
                                         }
                                     ]
                                    }
                             },
                        },
                        [
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 5
                                    }, # recursive search must for the para which has registed in parser
                                    "model": {
                                        "config":{"model_para1": 1}
                                    }
                                }
                            },
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 6
                                    }, # recursive search must for the para which has registed in parser
                                    "model": {
                                        "config":{"model_para1": 1}
                                    }
                                }
                            },
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 'a'
                                    }, # recursive search must for the para which has registed in parser
                                    "model": {
                                        "config":{"model_para1": 1}
                                    }
                                }
                            },
                        ]
                    ],
                ])
    def test_advance_recursive_search_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [
                        {
                            "_name": "base_config",
                            "imodel": {
                                "config": {},
                                "_search": {
                                    "config": [
                                        {
                                            "nest_para": "*@*",
                                            "nest_para_link": "*@*",
                                            "_search": {
                                                "nest_para": [5, 6],
                                            },
                                            "_link": {
                                                "nest_para": ['nest_para_link']
                                            }
                                        },
                                    ],
                                }
                             },
                        },
                        [
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 5,
                                        "nest_para_link": 5
                                    }, 
                                }
                            },
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 6,
                                        "nest_para_link": 6
                                    }, 
                                }
                            },
                        ]
                    ],
                    [
                        {
                            "_name": "base_config",
                            "imodel": {
                                "config": {},
                                "_search": {
                                    "config": [
                                        {
                                            "nest_para": "*@*",
                                            "nest_para_link": "*@*",
                                            "_search": {
                                                "nest_para": [5, 6],
                                            },
                                        },
                                    ],
                                },
                                "_link": {
                                    "config.nest_para": ['config.nest_para_link']
                                }
                             },
                        },
                        [
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 5,
                                        "nest_para_link": 5
                                    }, 
                                }
                            },
                            {
                                "_name": "base_config",
                                "imodel": {
                                    "config": {
                                        "nest_para": 6,
                                        "nest_para_link": 6
                                    }, 
                                }
                            },
                        ]
                    ],
                ])
    def test_link_search_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result

    @pytest.mark.parametrize(("config_dict", "result"), [ 
                    [ # Basic inherit
                        {
                            "embedding": {
                                "_base": "test_base_embedding_config",
                             },
                        },
                        [
                            {
                                "embedding": {
                                    "_name": 'test_base_embedding_config',
                                    "config": {
                                        "parent_para1": 1,
                                        "parent_para2": 2,
                                        "parent_para3": {
                                            "parent_para3_1": 31,
                                            "parent_para3_2": 32,
                                        },
                                    }
                                }, 
                            },
                        ]
                    ],
                    [ # Inherit & update the paras of parent
                        {
                            "embedding": {
                                "_base": "test_base_embedding_config",
                                "config": {
                                    "parent_para1": 'a',
                                    "parent_para3": {
                                        "parent_para3_1": "b",
                                        "parent_para3_3": "c"
                                    },
                                }
                             },
                        },
                        [
                            {
                                "embedding": {
                                    "_name": 'test_base_embedding_config',
                                    "config": {
                                        "parent_para1": 'a',
                                        "parent_para2": 2,
                                        "parent_para3": {
                                            "parent_para3_1": 'b',
                                            "parent_para3_2": 32,
                                            "parent_para3_3": 'c',
                                        },
                                    }
                                }, 
                            },
                        ]
                    ],
                ])
    def test_inherit_config(self, config_dict, result):
        config = BaseConfigParser(config_file=config_dict)
        parser_result = [json.dumps(one, indent=2, sort_keys=True) for one in config.parser()]
        parser_result.sort()
        result = [json.dumps(one, indent=2, sort_keys=True) for one in result]
        result.sort()
        assert parser_result == result
