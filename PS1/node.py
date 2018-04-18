class Node:
    def __init__(self):
        '''
        There are two types of node:
        brach node, which is actually the attribute in the tree
        leaf node, which is actually the label (classification) of the tree
        '''
        self.label = None   # leaf node
        self.branch = None  # branch node
        self.children = {}