class TreeNode:
    def __init__(self, value):
        self.value = value  # Node's value
        self.children = []  # List to store child nodes
    
    def add_child(self, child):
        self.children.append(child)  # Add a child node
    
    def display(self, prefix="", is_last=True):
        # Print the current node with pipes and branches
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.value)
        
        # Update the prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Recursively display children
        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            child.display(new_prefix, is_last_child)