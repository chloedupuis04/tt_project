

class IntervalTree:
    def __init__(self, lo, hi, parent=None, owner=None,y_tree=None):
        self.lo = lo
        self.hi = hi
        self.parent = parent
        self.children = []
        self.owner = owner
        self.y_tree = y_tree

    def add_child(self, child):
        self.children.append(child)
        return child
    
    def is_leaf(self):
        return len(self.children) == 0

    def leaves(self):
        if self.is_leaf():
            yield self
        else:
            for c in self.children:
                yield from c.leaves()

    def intervals(self):
        return [[leaf.lo, leaf.hi] for leaf in self.leaves()]

    def print_tree(self, prefix="", is_last=True, is_root=True):
        '''
        A nice way to visualize the tree
        '''
        label = "[{}, {}]".format(self.lo, self.hi)
        if self.is_leaf():
            label += "  (leaf)"
        if is_root:
            print(label)
        else:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + label)
            prefix += "    " if is_last else "│   "
        for i, child in enumerate(self.children):
            child.print_tree(prefix, is_last=(i == len(self.children) - 1),
                             is_root=False)
