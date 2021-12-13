import os

HOME = os.environ['HOME']

def get_root():
    """TODO: Docstring for get_root.
    :returns: TODO

    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def check_cur_is_root(cur_dir):
        """TODO: Docstring for check_cur_is_root.

        :arg1: TODO
        :returns: TODO

        """
        root_sign_list = ['.git', '.root', 'requirements.txt', 'requirement.txt']
        for root_sign in root_sign_list:
            if os.path.exists(os.path.join(cur_dir, root_sign)):
                return True
        return False

    while cur_dir != HOME:
        if check_cur_is_root(cur_dir):
            break
        cur_dir = os.path.dirname(cur_dir)

    if cur_dir == HOME:
        return '.'
    else:
        return cur_dir
