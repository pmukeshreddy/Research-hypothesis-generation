import networkx as nx
import numpy as np
import re
import json
from datetime import datetime
from dateutil import parser as date_parser
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from dateutil import parser


class ConceptTemporalGraph:
    def __init__(self, collector):
        """
        Initialize the concept temporal graph with a paper collector
        
        Args:
            collector: An instance that contains papers with publication dates
        """
        self.collector = collector
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges between nodes
        
        # For tracking nodes by type
        self.nodes_by_type = defaultdict(list)
        
        # For tracking concept frequency
        self.concept_mentions = Counter()
        
        # Domain-specific concepts for neuroscience (customize for your domain)
        self.domain_concepts = {
            "brain_region": [
                "prefrontal cortex", "hippocampus", "amygdala", "cerebellum",
                "basal ganglia", "thalamus", "hypothalamus", "brainstem", 
                "neocortex", "striatum", "substantia nigra", "parietal lobe",
                "temporal lobe", "frontal lobe", "occipital lobe", "insula"
            ],
            "method": [
                "fMRI", "EEG", "MEG", "PET", "optogenetics", "electrophysiology",
                "calcium imaging", "CLARITY", "deep brain stimulation", "TMS",
                "tDCS", "single-cell recording", "patch clamp", "two-photon microscopy"
            ],
            "cognitive_process": [
                "attention", "memory", "learning", "decision-making", "perception",
                "executive function", "emotion", "language", "consciousness", 
                "social cognition", "spatial navigation", "reward processing"
            ],
            "condition": [
                "Alzheimer's", "Parkinson's", "schizophrenia", "depression", "anxiety",
                "autism", "ADHD", "epilepsy", "stroke", "traumatic brain injury",
                "multiple sclerosis", "dementia", "OCD", "PTSD"
            ],
            "cellular_component": [
                "neuron", "glia", "astrocyte", "oligodendrocyte", "microglia",
                "synapse", "dendrite", "axon", "myelin", "neurotransmitter",
                "receptor", "ion channel", "action potential", "long-term potentiation"
            ]
        }
    
    def extract_concepts_from_text(self, text, paper_id):
        """
        Extract domain-specific concepts from text using pattern matching
        
        Args:
            text: The text to analyze
            paper_id: ID of the paper the text is from
            
        Returns:
            List of concept dictionaries
        """
        concepts = []
        
        # Pattern matching for domain-specific concepts
        for concept_type, concept_list in self.domain_concepts.items():
            for concept in concept_list:
                # Use word boundary regex for more accurate matching
                matches = re.findall(r'\b' + re.escape(concept.lower()) + r'\b', text.lower())
                if re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()):
                    concepts.append({
                        "id": f"concept_{concept.lower().replace(' ', '_')}",
                        "name": concept,
                        "type": concept_type,
                        "source": "pattern",
                        "paper_id": paper_id,
                        "frequency": len(matches) 
                    })
                    
                    # Track concept mentions
                    self.concept_mentions[concept] += len(matches)

        
        return concepts
    
    def add_paper_node(self, paper):
        """
        Add a paper node to the graph with temporal information
        
        Args:
            paper: Paper dictionary from the collector
            
        Returns:
            The node ID
        """
        # Parse publication date for temporal ordering

        
        year = None
        if paper.get('publication_date'):
            try:
                date_obj = date_parser.parse(paper['publication_date'])
                year = date_obj.year
             #   print(year)
            except:
                print("did you got reason")
        
        # Create unique ID
        node_id = f"paper_{paper['id']}"
        
        # Add node with all paper properties
        self.graph.add_node(
            node_id,
            type="paper",
            title=paper['title'],
            abstract=paper['abstract'],
            publication_date=paper.get('publication_date', ''),
            year=year,
            source=paper['source']
        )
        
        # Track by type
        self.nodes_by_type["paper"].append(node_id)
        
        return node_id
    
    def add_concept_node(self, concept):
        """
        Add a concept node to the graph
        
        Args:
            concept: Concept dictionary
            
        Returns:
            The node ID
        """
        # Check if node already exists
        if self.graph.has_node(concept["id"]):
            return concept["id"]
        
        # Add the concept node
        self.graph.add_node(
            concept["id"],
            type="concept",
            name=concept["name"],
            concept_type=concept["type"],
            source=concept["source"]
        )
        
        # Track by type
        self.nodes_by_type["concept"].append(concept["id"])
        
        return concept["id"]
    
    def connect_concepts_to_papers(self, paper_id, concepts):
        """
        Connect concepts to a paper
        
        Args:
            paper_id: The paper node ID
            concepts: List of concept dictionaries
        """
        for concept in concepts:
            # Make sure concept node exists
            concept_id = self.add_concept_node(concept)
            
            # Add relationship
            self.graph.add_edge(
                paper_id, concept_id,
                type="MENTIONS",
                timestamp=self.graph.nodes[paper_id].get("year"),
                frequency=concept.get("frequency", 1)
            )
    
    def connect_papers_temporally(self, paper_ids_with_dates):
        """
        Create temporal connections between papers based on publication dates
        
        Args:
            paper_ids_with_dates: List of tuples (paper_id, date)
        """
        # Sort by date
        paper_ids_with_dates.sort(key=lambda x: x[1])
        
        # Create PRECEDES relationships in temporal sequence
        for i in range(len(paper_ids_with_dates) - 1):
            id1, date1 = paper_ids_with_dates[i]
            id2, date2 = paper_ids_with_dates[i + 1]
            
            # Add directed edge from earlier to later paper
            self.graph.add_edge(
                id1, id2,
                type="PRECEDES",
                time_gap_days=(date2 - date1).days
            )
    
    def analyze_concept_evolution(self):
        """
        Analyze how concepts evolve over time by examining temporal patterns
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            "concept_timeline": {},
            "research_span": {},
            "emerging_concepts": [],
            "consistent_concepts": []
        }
        
        # Get all concept nodes
        concept_nodes = self.nodes_by_type["concept"]
        
        for concept_id in concept_nodes:
            # Get papers that mention this concept
            paper_mentions = []
            
            for paper_id in self.nodes_by_type["paper"]:
                if self.graph.has_edge(paper_id, concept_id):
                    # Get paper year if available
                    year = self.graph.nodes[paper_id].get("year")
                    if year:
                        paper_mentions.append((paper_id, year))
            
            # Sort by year
            paper_mentions.sort(key=lambda x: x[1])
            
            if paper_mentions:
                # Get concept name
                concept_name = self.graph.nodes[concept_id].get("name")
                
                # Store timeline
                results["concept_timeline"][concept_name] = [
                    {"paper": p, "year": y} for p, y in paper_mentions
                ]
                
                # Calculate research span
                first_year = paper_mentions[0][1]
                last_year = paper_mentions[-1][1]
                span = last_year - first_year
                
                results["research_span"][concept_name] = {
                    "first_year": first_year,
                    "last_year": last_year,
                    "span": span
                }
                
                # Identify emerging vs. consistent concepts
                if span >= 5 and len(paper_mentions) > 10:
                    results["consistent_concepts"].append(concept_name)
                elif len(paper_mentions) >= 2 and last_year >= datetime.now().year - 3:
                    results["emerging_concepts"].append(concept_name)
        
        return results
    
    def build_temporal_graph(self):
        """
        Build the concept temporal graph from the collector's papers
        """
        print("Building temporal concept graph...")
        
        # Store paper dates for temporal connections
        paper_dates = []
        
        # Process each paper
        for paper in tqdm(self.collector.papers, desc="Processing papers"):
            # Add paper node
            paper_node_id = self.add_paper_node(paper)
            
            # Extract and add concepts from title and abstract
            text_to_analyze = f"{paper['title']} {paper['abstract']}"
            concepts = self.extract_concepts_from_text(text_to_analyze, paper['id'])
            self.connect_concepts_to_papers(paper_node_id, concepts)
            
            # Add full text analysis if available
            if paper.get('full_text'):
                # Extract concepts from full text
                full_text_concepts = self.extract_concepts_from_text(
                    paper['full_text'][:5000], paper['id'])
                self.connect_concepts_to_papers(paper_node_id, full_text_concepts)
            
            # Store publication date for temporal connections
            if paper.get('publication_date'):
                try:
                    date_obj = date_parser.parse(paper['publication_date'])
                    paper_dates.append((paper_node_id, date_obj))
                except:
                    pass
        
        # Create temporal connections
        if paper_dates:
            print("Creating temporal connections...")
            self.connect_papers_temporally(paper_dates)
        
        print(f"Temporal graph construction complete with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
    


    def get_concept_frequency_by_month(self):
        freq = defaultdict(lambda: defaultdict(int))
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") != "MENTIONS":
                continue
            pub_date = self.graph.nodes[u].get("publication_date")
            if not pub_date:
                continue
            dt = parser.parse(pub_date)
            bucket = dt.strftime("%Y-%m")           # e.g. "2023-04"
            concept = self.graph.nodes[v]["name"]
            freq[bucket][concept] += data.get("frequency", 1)
        return freq
        
    def get_concept_frequency_by_year(self):
        concept_freq = defaultdict(lambda: defaultdict(int))
        # iterate every MENTIONS edge exactly once
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") != "MENTIONS":
                continue
            year = self.graph.nodes[u].get("year")
            if not year:
                continue
            # how many matches did we record on that edge?
            freq = data.get("frequency", 1)
            concept_name = self.graph.nodes[v]["name"]
            concept_freq[year][concept_name] += freq
        return concept_freq


    
    def visualize_concept_timeline(self, concept_name=None, filename="concept_timeline.png"):
        """
        Visualize concept frequency over time
        
        Args:
            concept_name: Specific concept to visualize (None for top concepts)
            filename: Output filename
        """
        # Get concept frequency by year
        freq_by_year = self.get_concept_frequency_by_month()


    
    # ADD DEBUG CODE HERE ↓↓↓
    # Debug: Print raw concept frequencies
        print("\nDEBUG - Raw concept mention counts:")
        for concept, count in self.concept_mentions.most_common(10):
            print(f"  {concept}: {count}")

    # Debug: Check frequency by year data
        print("\nDEBUG - Frequency by year:")
        for year in sorted(freq_by_year.keys()):
            print(f"Year {year}:")
            counts = list(freq_by_year[year].values())
            concepts = list(freq_by_year[year].keys())
            for i in range(min(5, len(concepts))):
                print(f"  {concepts[i]}: {counts[i]}")
        
        # Check if all values are the same
            if counts and all(x == counts[0] for x in counts):
                print(f"  ⚠️ All {len(counts)} concepts have identical frequency: {counts[0]}")
            else:
                print(f"  ✅ Concepts have different frequencies (min: {min(counts) if counts else 0}, max: {max(counts) if counts else 0})")

        
        # Prepare data for plotting
        if concept_name:
            # Plot timeline for a specific concept
            years = []
            frequencies = []
            
            for year in sorted(freq_by_year.keys()):
                if concept_name in freq_by_year[year]:
                    years.append(year)
                    frequencies.append(freq_by_year[year][concept_name])
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.bar(years, frequencies, color='royalblue')
            plt.title(f"Timeline for Concept: {concept_name}")
            plt.xlabel("Year")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
        else:
            # Plot timeline for top 5 concepts
            top_concepts = self.concept_mentions.most_common(5)
            
            # Get data for each concept
            concept_data = {}
            for concept, _ in top_concepts:
                concept_data[concept] = []
            
            # Get years range
            all_years = sorted(freq_by_year.keys())
            
            # Fill in data for each year
            for year in all_years:
                for concept, _ in top_concepts:
                    concept_data[concept].append(freq_by_year[year].get(concept, 0))
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(all_years))
            width = 0.15
            offset = 0
            
            for concept, values in concept_data.items():
                plt.bar(x + offset, values, width, label=concept)
                offset += width
            
            plt.xlabel("Year")
            plt.ylabel("Frequency")
            plt.title("Concept Evolution Over Time")
            plt.xticks(x + width * len(concept_data) / 2 - width / 2, all_years)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Concept timeline visualization saved to {filename}")
        
        return plt
    
    def export_to_json(self, filename="concept_temporal_graph.json"):
        """
        Export the graph to JSON format for storage or transfer
        
        Args:
            filename: Output filename
        """
        # Convert graph to dictionary
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Export nodes
        for node_id, data in self.graph.nodes(data=True):
            node_data = {"id": node_id}
            node_data.update(data)
            graph_data["nodes"].append(node_data)
        
        # Export edges
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            edge_data = {
                "source": u,
                "target": v,
                "key": key
            }
            edge_data.update(data)
            graph_data["edges"].append(edge_data)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph exported to {filename}")
        
        return graph_data
    def debug_concept_frequency(self):
        """Debug function to identify issues with concept frequency tracking"""
        print("\n=== DEBUGGING CONCEPT FREQUENCY ===")
        
        # 1. Check if we have papers with valid years
        paper_years = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'paper':
                year = data.get('year')
                paper_years[node_id] = year
        
        print(f"Total papers: {len(self.nodes_by_type['paper'])}")
        print(f"Papers with valid years: {sum(1 for y in paper_years.values() if y is not None)}")
        print(f"Unique years found: {sorted(set(y for y in paper_years.values() if y is not None))}")
        
        # 2. Check if we have MENTIONS edges
        mentions_edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if data.get('type') == 'MENTIONS':
                mentions_edges.append((u, v, k, data))
        
        print(f"\nTotal edges: {self.graph.number_of_edges()}")
        print(f"MENTIONS edges: {len(mentions_edges)}")
        
        # 3. Check frequency distribution on MENTIONS edges
        if mentions_edges:
            frequencies = [data.get('frequency', 1) for _, _, _, data in mentions_edges]
            print(f"Frequency values: min={min(frequencies)}, max={max(frequencies)}, avg={sum(frequencies)/len(frequencies):.2f}")
            print(f"All frequencies equal: {all(f == frequencies[0] for f in frequencies)}")
            counter = Counter(frequencies)
            print(f"Frequency distribution: {dict(counter.most_common(5))}")
        
        # 4. Check specific examples of concept mentions
        print("\nSample MENTIONS edges:")
        for i, (u, v, k, data) in enumerate(mentions_edges[:5]):
            paper_year = self.graph.nodes[u].get('year')
            paper_title = self.graph.nodes[u].get('title', 'Unknown')[:30] + '...'
            concept_name = self.graph.nodes[v].get('name', 'Unknown')
            print(f"  {i+1}. Paper '{paper_title}' ({paper_year}) -> Concept '{concept_name}' [freq: {data.get('frequency', 1)}]")
        
        # 5. Test the get_concept_frequency_by_year method directly
        freq_by_year = self.get_concept_frequency_by_year()
        print(f"\nYears in frequency data: {sorted(freq_by_year.keys())}")
        
        # Check the raw data structure
       #
        # 6. Check regex pattern matching in the extract_concepts_from_text method
        print("\nTesting concept extraction:")
        test_text = "This is a test about the hippocampus and fMRI studies of memory."
        test_concepts = self.extract_concepts_from_text(test_text, "test_paper")
        print(f"Found {len(test_concepts)} concepts in test text")
        for concept in test_concepts:
            print(f"  {concept['name']} (type: {concept['type']}, frequency: {concept['frequency']})")
        
        print("\n=== END DEBUGGING ===")
