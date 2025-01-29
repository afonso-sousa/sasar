import random

from .utils.synonymy import find_antonym, find_synonym


class TransformationsMixin:
    def find_node_with_arg0_arg1_relations(self):
        for i in range(len(self.nodes)):
            arg0_relations = []
            arg1_relations = []
            
            # Collect all ARG0 and ARG1 relations for the current node
            for edge in self.edges:
                if edge[0] == i:
                    if edge[2] == "ARG0":
                        arg0_relations.append(edge)
                    elif edge[2] == "ARG1":
                        arg1_relations.append(edge)
            
            # Check if the node has relations with both ARG0 and ARG1 nodes
            if arg0_relations and arg1_relations:
                arg0_edge = random.choice(arg0_relations)
                arg1_edge = random.choice(arg1_relations)
                return i, arg0_edge, arg1_edge

        return None

    def change_of_voice(self):
        out = self.find_node_with_arg0_arg1_relations()
        if not out:
            return False
        _, arg0_edge, arg1_edge = out

        new_subject_edge = (arg1_edge[0], arg1_edge[1], "ARG0")
        new_agent_edge = (arg0_edge[0], arg0_edge[1], "by")
        
        self.edges.remove(arg0_edge)
        self.edges.remove(arg1_edge)
        self.edges.append(new_subject_edge)
        self.edges.append(new_agent_edge)

        return True

    def converse_substitution(self):
        out = self.find_node_with_arg0_arg1_relations()
        if not out:
            return False
        i, arg0_edge, arg1_edge = out
        if len(self.nodes[i].word) > 1:
            return False
        verb_ant = find_antonym(self.nodes[i].word[0])
        if not verb_ant:
            return False

        new_subject_edge = (arg1_edge[0], arg1_edge[1], "ARG0")
        new_agent_edge = (arg0_edge[0], arg0_edge[1], "to")
        
        self.edges.remove(arg0_edge)
        self.edges.remove(arg1_edge)
        self.edges.append(new_subject_edge)
        self.edges.append(new_agent_edge)

        self.nodes[i].word = [verb_ant]

        return True

    def synonym_substitution(self):
        count_syns = 0
        for i, node in enumerate(self.nodes):
            if len(node.word) > 1:
                continue

            syn = find_synonym(node.word[0])
            if syn:
                self.nodes[i].word = [syn]
                count_syns += 1
        
        return count_syns
