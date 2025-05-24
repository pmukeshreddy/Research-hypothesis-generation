import json
import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging
from collections import defaultdict


class PatternRecognitionAgent(PatternRecognitionAgent):
    def __init__(self, concept_graph=None, llm_api_key=None, llm_endpoint=None, llm_model="gpt-4"):
        # Call the parent class's __init__ method for proper initialization
        super().__init__(concept_graph)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Add LLM-specific attributes
        self.llm_api_key = llm_api_key
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        
        # Set concept_graph
        self.concept_graph = concept_graph
        
        # For storing llm enhanced results
        self.enhanced_patterns = {
            "concept_relationships": [],
            "semantic_clusters": [],
            "emerging_themes": [],
            "research_gaps": [],
            "interdisciplinary_connections": []
        }
        
        # Storing explanations and interpretations
        self.interpretations = {
            "trend_explanations": {},
            "trajectory_interpretations": {},
            "impact_assessments": {},
            "research_narratives": {}
        }
        
        # Confidence score 
        self.confidence_scores = {
            "traditional_metrics": {},
            "llm_assessments": {},
            "combined_confidence": {}
        }
            
    def _call_google_api(self, prompt, temperature=0.0, max_tokens=None):
        """Call Google AI API (specifically Gemini)"""
        import requests
        import json
    
        try:
            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 1024
            
            # For Gemini API, we need to format the request differently
            api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
            
            # Add API key to URL as a query parameter
            if "?" not in api_url:
                api_url += f"?key={self.llm_api_key}"
            else:
                api_url += f"&key={self.llm_api_key}"
            
            # Prepare API request headers
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare request payload specifically for Gemini
            data = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
    "maxOutputTokens": max_tokens or 1024,
    "topP": 0.9,
    "topK": 40
                }
            }
        
        # Make the API call
            response = requests.post(api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Extract the generated text from the Gemini response format
                try:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    self.logger.error(f"Unexpected Gemini API response format: {response_json}")
                    return "{}"
            else:
                self.logger.error(f"Google API error: {response.status_code} - {response.text}")
                return "{}"
        except Exception as e:
            self.logger.error(f"Error calling Google API: {str(e)}")
            return "{}"          
    def _call_llm_api(self, prompt, temperature=0.0, max_tokens=None):
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: String prompt to send to the LLM API
            temperature: Temperature parameter for generation (default: 0.0)
            max_tokens: Maximum tokens to generate (default: None)
            
        Returns:
            String response from the LLM API
        """
        self.logger.info("Calling LLM API with prompt of length: %d", len(prompt))
        
        # Check which API provider to use
        if self.llm_endpoint and "openai" in self.llm_endpoint.lower():
            return self._call_openai_api(prompt, temperature, max_tokens)
        elif self.llm_endpoint and ("google" in self.llm_endpoint.lower() or 
                                    "gemini" in self.llm_endpoint.lower() or 
                                    "palm" in self.llm_endpoint.lower() or
                                    "vertex" in self.llm_endpoint.lower()):
            return self._call_google_api(prompt, temperature, max_tokens)
        elif self.llm_endpoint and "anthropic" in self.llm_endpoint.lower():
            return self._call_anthropic_api(prompt, temperature, max_tokens)
        elif self.llm_endpoint and "huggingface" in self.llm_endpoint.lower():
            return self._call_huggingface_api(prompt, temperature, max_tokens)
        elif self.llm_endpoint and "cohere" in self.llm_endpoint.lower():
            return self._call_cohere_api(prompt, temperature, max_tokens)
        else:
            self.logger.warning("No specific API provider detected. Defaulting to OpenAI.")
            return self._call_openai_api(prompt, temperature, max_tokens)

    def _call_openai_api(self, prompt, temperature=0.0, max_tokens=None):
        """Call OpenAI API"""
        import openai
        import json
        
        try:
            openai.api_key = self.llm_api_key
            
            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 1024
                
            # Call the API based on the model specified
            if "gpt-4" in self.llm_model or "gpt-3.5" in self.llm_model:
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a scientific research analysis assistant that responds in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                # For older models that use the Completion API
                response = openai.Completion.create(
                    engine=self.llm_model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].text
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return "{}"  # Return empty JSON object on error

    def _call_anthropic_api(self, prompt, temperature=0.0, max_tokens=None):
        """Call Anthropic Claude API"""
        import requests
        import json
        
        try:
            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 1024
                
            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.llm_api_key
            }
            
            data = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "model": self.llm_model,
                "temperature": temperature,
                "max_tokens_to_sample": max_tokens
            }
            
            # Make the API call
            response = requests.post(
                self.llm_endpoint or "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json().get("completion", "{}")
            else:
                self.logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return "{}"
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            return "{}"
    
    def _call_huggingface_api(self, prompt, temperature=0.0, max_tokens=None):
        """Call Hugging Face Inference API"""
        import requests
        import json
        
        try:
            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 1024
                
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            }
            
            # Make the API call to the Hugging Face Inference API
            api_url = self.llm_endpoint or f"https://api-inference.huggingface.co/models/{self.llm_model}"
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the generated text from the response
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "{}")
                return result.get("generated_text", "{}")
            else:
                self.logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                return "{}"
        except Exception as e:
            self.logger.error(f"Error calling Hugging Face API: {str(e)}")
            return "{}"
        
    def _call_cohere_api(self, prompt, temperature=0.0, max_tokens=None):
        """Call Cohere API"""
        import requests
        import json
        
        try:
            # Set default max tokens if not provided
            if max_tokens is None:
                max_tokens = 1024
                
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "prompt": prompt,
                "model": self.llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "return_likelihoods": "NONE"
            }
            
            # Make the API call
            response = requests.post(
                self.llm_endpoint or "https://api.cohere.ai/v1/generate",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json().get("generations", [{}])[0].get("text", "{}")
            else:
                self.logger.error(f"Cohere API error: {response.status_code} - {response.text}")
                return "{}"
        except Exception as e:
            self.logger.error(f"Error calling Cohere API: {str(e)}")
            return "{}"

    def enhance_concept_extraction(self, text, paper_id=None):
        # This better understand the concepts (LLMs extract concepts)
        traditional_concepts = []
        if self.concept_graph:
            traditional_concepts = self.concept_graph.extract_concepts_from_text(text, paper_id)

        domain_concept_str = ""
        if hasattr(self.concept_graph, "domain_concepts"):
            domain_concept_str = "Some important domain concepts include:\n"
            for concept_type, concepts in self.concept_graph.domain_concepts.items():
                domain_concept_str += f"- {concept_type}: {', '.join(concepts[:5])}...\n"
        
        prompt = f"""
        You are an expert in scientific research analysis with deep knowledge of neuroscience and related fields.
        Extract important concepts from the following text, focusing on methods, technologies, brain regions, 
        conditions, and cognitive processes.
        For each concept, provide:
        1. The concept name
        2. The concept type (method, technology, brain_region, condition, cognitive_process, cellular_component, or other)
        3. A confidence score (0-1)
        Also identify relationships between concepts, including:
        1. Source concept
        2. Target concept
        3. Relationship type (e.g., "enables", "part_of", "applied_to", "associated_with")
        4. Confidence score (0-1)
        
        {domain_concept_str}
        
        Provide your response as a JSON object with "extracted_concepts" and "relationships" lists.
        
        TEXT TO ANALYZE:
        {text[:3000]}
        """
        llm_response = self._call_llm_api(prompt)
        enhanced_results = {"extracted_concepts": [], "relationships": []}
        if llm_response:
            try:
                llm_data = json.loads(llm_response)
                enhanced_results = llm_data
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON")
                # Try to extract JSON using regex if direct parsing fails
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                match = re.search(json_pattern, llm_response)
                if match:
                    try:
                        enhanced_results = json.loads(match.group(0))
                    except:
                        self.logger.error("Failed to extract JSON from LLM response")

            # Combine traditional and llm enhanced concepts
            combined_concepts = []
            # Let's start with traditional concepts
            concept_names = set()
            for concept in traditional_concepts:
                concept_id = concept.get("id", "")
                name = concept.get("name", "")
                concept_names.add(name.lower())

                combined_concepts.append({
                    "id": concept_id,
                    "name": name,
                    "type": concept.get("type", ""),
                    "source": "traditional",
                    "paper_id": paper_id,
                    "confidence": 1.0
                })
            # Add llm extracted concepts
            for concept in enhanced_results.get("extracted_concepts", []):
                name = concept.get("concept", "")
                if name.lower() not in concept_names:  # Fixed: check against concept_names set
                    concept_id = f"concept_{name.lower().replace(' ', '_')}"
                    concept_type = concept.get("type", "other")
                    confidence = concept.get("confidence", 0.8)

                    combined_concepts.append(
                        {
                            "id": concept_id,
                            "name": name,
                            "type": concept_type,
                            "source": "llm",
                            "paper_id": paper_id,
                            "confidence": confidence
                        }
                    )

                    concept_names.add(name.lower())
        # Store relationships for later analysis
        if paper_id and enhanced_results.get("relationships"):
            for rel in enhanced_results.get("relationships", []):
                self.enhanced_patterns["concept_relationships"].append({
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "type": rel.get("type", "associated_with"),   
                    "confidence": rel.get("confidence", 0.7),
                    "paper_id": paper_id
                })

        return combined_concepts

    def identify_trending_concepts(self, time_window=3, min_frequency=2):
        # First use traditional methods to identify trending concepts
        traditional_trending = super().identify_trending_concepts(time_window, min_frequency)
        # Enhance top trending concepts with llm analysis
        enhanced_trending = []
    #    print("gi")
   #     print(traditional_trending)

        for i, concept_data in enumerate(traditional_trending[:10]):
            concept = concept_data["concept"]
            growth_rate = concept_data["growth_rate"]
            recent_freq = concept_data["recent_freq"]
            earlier_freq = concept_data["earlier_freq"]

            prompt = f"""
            Analyze the following trending research concept:
            
            Concept: {concept}
            Growth Rate: {growth_rate:.2f}
            Recent Frequency: {recent_freq:.2f} (past {time_window} years)
            Earlier Frequency: {earlier_freq:.2f}
            
            Based on your knowledge of scientific research trends and this data:
            1. What might be driving this trend? Consider technological, social, or scientific factors.
            2. What barriers or challenges might affect continued growth?
            3. What is the broader context or significance of this concept's growth?
            4. What potential impact might this concept have on the field?
            
            Provide your analysis as a JSON object with "drivers", "barriers", "context", and "potential_impact" fields.
            """
          #  print("fdf")
         #   print(prompt)
            llm_response = self._call_llm_api(prompt)
            trend_analysis = {}

            if llm_response:
                try:
                    trend_analysis = json.loads(llm_response)
                    if isinstance(trend_analysis, dict) and "trend_analysis" in trend_analysis:
                        trend_analysis = trend_analysis.get("trend_analysis", {})
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.error(f"Error parsing LLM trend analysis: {str(e)}")
                    # Try to extract JSON using regex if direct parsing fails
                    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                    match = re.search(json_pattern, llm_response)
                    if match:
                        try:
                            trend_analysis = json.loads(match.group(0))
                            if isinstance(trend_analysis, dict) and "trend_analysis" in trend_analysis:
                                trend_analysis = trend_analysis.get("trend_analysis", {})
                        except:
                            self.logger.error("Failed to extract JSON from LLM response")

            enhanced_concept = concept_data.copy()
            enhanced_concept.update({
                "drivers": trend_analysis.get("drivers", []),
                "barriers": trend_analysis.get("barriers", []),
                "context": trend_analysis.get("context", ""),
                "potential_impact": trend_analysis.get("potential_impact", "")
            })
           # print(enhanced_concept)
            enhanced_trending.append(enhanced_concept)
            self.interpretations["trend_explanations"][concept] = trend_analysis
            
        # Concepts not analyzed by llm, add without enhancement    
        for concept_data in traditional_trending[10:]:
            enhanced_trending.append(concept_data)

        self.identified_patterns["trending_concepts"] = enhanced_trending

        return enhanced_trending
    
    def identify_semantic_cluster(self, min_samples=2, eps=0.5):
        # Fixed: call the parent class method with the correct name
        traditional_clusters = super().identify_concept_cluster(min_samples, eps)

        enhanced_clusters = []

        for cluster in traditional_clusters[:5]:
            cluster_id = cluster["cluster_id"]
            concepts = cluster["concepts"]

            prompt = f"""
            Analyze the following cluster of research concepts that frequently appear together:
            
            Concepts: {', '.join(concepts)}
            
            Based on your knowledge of scientific research:
            1. What unifying themes or research questions connect these concepts?
            2. What are the potential applications or implications of research in this cluster?
            3. Identify any sub-groups or hierarchical relationships within these concepts.
            4. Suggest a descriptive name for this research area.
            
            Provide your analysis as a JSON object with fields for "unifying_themes", "potential_applications", 
            "concept_relationships", and "suggested_name".
            """

            llm_response = self._call_llm_api(prompt)
            semantic_analysis = {}

            if llm_response:
                try:
                    semantic_analysis = json.loads(llm_response)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse LLM response as JSON for semantic analysis")
                    # Try to extract JSON using regex
                    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                    match = re.search(json_pattern, llm_response)
                    if match:
                        try:
                            semantic_analysis = json.loads(match.group(0))
                        except:
                            self.logger.error("Failed to extract JSON from LLM response")

                enhanced_cluster = cluster.copy()
                enhanced_cluster.update({
                    "unifying_themes": semantic_analysis.get("unifying_themes", []),
                    "potential_applications": semantic_analysis.get("potential_applications", []),
                    "concept_relationships": semantic_analysis.get("concept_relationships", []),
                    "suggested_name": semantic_analysis.get("suggested_name", "")
                })
                enhanced_clusters.append(enhanced_cluster)

        for cluster in traditional_clusters[5:]:
            enhanced_clusters.append(cluster)

        self.enhanced_patterns["semantic_clusters"] = enhanced_clusters
        return enhanced_clusters
        
    def identify_research_gaps(self):
        research_data = {
            "trending_concepts": self.identified_patterns.get("trending_concepts", []),
            "concept_clusters": self.identified_patterns.get("concept_clusters", [])  # Fixed: using correct key
        }
        # Prepare context for llm
        trending_concepts_str = "\n".join([
            f"- {c['concept']} (Growth Rate: {c['growth_rate']:.2f})" 
            for c in research_data["trending_concepts"][:10]
        ])
        cluster_str = "\n".join([
            f"- Cluster {c['cluster_id']}: {', '.join(c['concepts'][:5])}" + 
            ("..." if len(c['concepts']) > 5 else "")
            for c in research_data["concept_clusters"][:5]  # Fixed: using correct key
        ])
        
        prompt = f"""
        You are an expert in scientific research analysis with deep knowledge of neuroscience and related fields.
        Based on the following data about research trends and concept clusters, identify potential research gaps
        and opportunities.
        
        Top Trending Concepts:
        {trending_concepts_str}
        
        Concept Clusters:
        {cluster_str}
        
        For each identified research gap:
        1. Provide a clear description of the gap
        2. Suggest 2-3 potential research directions to address this gap
        3. Explain why this gap is significant
        4. Rate the potential impact of addressing this gap (0-1)
        
        Identify at least 3 but no more than 5 significant research gaps.
        Provide your analysis as a JSON array of objects with "description", "potential_directions", "significance", 
        and "impact_score" fields.
        """
        
        llm_response = self._call_llm_api(prompt)
        research_gaps = []

        if llm_response:
            try:
                research_gaps = json.loads(llm_response)
                if isinstance(research_gaps, dict) and "research_gaps" in research_gaps:
                    research_gaps = research_gaps["research_gaps"]
            except (json.JSONDecodeError, TypeError):
                self.logger.error("Failed to parse LLM response as JSON for research gaps")
                # Try to extract JSON using regex
                json_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'
                match = re.search(json_pattern, llm_response)
                if match:
                    try:
                        research_gaps = json.loads(match.group(0))
                    except:
                        self.logger.error("Failed to extract JSON array from LLM response")
        
        self.enhanced_patterns["research_gaps"] = research_gaps
        return research_gaps
    
    def identify_interdisciplinary_connections(self):
        clusters = self.identified_patterns.get("concept_clusters", [])
        if not clusters:
            clusters = self.identify_concept_cluster()

        if len(clusters) < 2:
            return []
        
        cluster_str = "\n".join([
            f"Area {i+1}: {', '.join(c['concepts'][:7])}" + 
            ("..." if len(c['concepts']) > 7 else "")
            for i, c in enumerate(clusters[:5])
        ])
        
        prompt = f"""
        You are an expert in scientific research analysis with broad interdisciplinary knowledge.
        Based on the following research areas, identify potential interdisciplinary connections and
        collaboration opportunities.
        
        Research Areas:
        {cluster_str}
        
        For each potential interdisciplinary connection:
        1. Identify which research areas could connect (e.g., "Area 1 and Area 3")
        2. Describe the potential connection or overlap
        3. Suggest specific interdisciplinary research questions
        4. Explain potential benefits of this interdisciplinary approach
        5. Identify any emerging fields that might result from this connection
        
        Identify at least 3 interesting interdisciplinary connections.
        Provide your analysis as a JSON array of objects with fields for the connections.
        """
        llm_response = self._call_llm_api(prompt)
        connections = []

        if llm_response:
            try:
                connections = json.loads(llm_response)
                if isinstance(connections, dict) and "connections" in connections:
                    connections = connections["connections"]
            except (json.JSONDecodeError, TypeError):
                self.logger.error("Failed to parse LLM response as JSON for interdisciplinary connections")
                # Try to extract JSON using regex
                json_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'
                match = re.search(json_pattern, llm_response)
                if match:
                    try:
                        connections = json.loads(match.group(0))
                    except:
                        self.logger.error("Failed to extract JSON array from LLM response")

        self.enhanced_patterns["interdisciplinary_connections"] = connections
        return connections
    
    def predict_concept_growth_with_context(self, prediction_years=3):
        # First predict concept growth through traditional method
        traditional_predictions = self.predict_concept_growth(prediction_years)

        # Get concepts for enhancements
        fast_growing = traditional_predictions.get("fast_growing", [])
        stable = traditional_predictions.get("stable", [])
        declining = traditional_predictions.get("declining", [])

        all_top_concepts = (
            fast_growing[:5] + stable[:3] + declining[:2]
        )

        for concept_data in all_top_concepts:
            concept = concept_data["concept"]
            growth_rate = concept_data["growth_rate"]

            # Skip if we already have explanation for this concept
            if concept in self.interpretations["trend_explanations"]:
                continue

            prompt = f"""
            You are a scientific research trend analyst with expertise in predicting research trajectories.
            
            Concept: {concept}
            Predicted Growth Rate: {growth_rate:.2f} over the next {prediction_years} years
            Current Standing: {"Fast Growing" if growth_rate > 0.2 else "Stable" if growth_rate >= -0.1 else "Declining"}
            
            Based on your knowledge of research trends and broader technological/scientific developments:
            1. What external factors might influence this concept's growth trajectory?
            2. What potential breakthroughs could accelerate or decelerate this trend?
            3. How might this concept interact with other emerging research areas?
            4. What is your confidence level in this prediction (0-100%)? Why?
            
            Provide your analysis as a JSON object with fields for "external_factors", "potential_breakthroughs", 
            "interactions", "confidence_level", and "reasoning".
            """
            context_analysis = {}
            llm_response = self._call_llm_api(prompt)

            if llm_response:
                try:
                    context_analysis = json.loads(llm_response)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse LLM response as JSON for contextual analysis")
                    # Try to extract JSON using regex
                    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                    match = re.search(json_pattern, llm_response)
                    if match:
                        try:
                            context_analysis = json.loads(match.group(0))
                        except:
                            self.logger.error("Failed to extract JSON from LLM response")
            
            self.interpretations["trend_explanations"][concept] = context_analysis

            # Update confidence score
            if "confidence_level" in context_analysis:
                try:
                    confidence = float(context_analysis["confidence_level"]) / 100
                    self.confidence_scores["llm_assessments"][concept] = confidence
                    traditional_conf = 0.7  # Default confidence for traditional methods
                    self.confidence_scores["traditional_metrics"][concept] = traditional_conf
                    self.confidence_scores["combined_confidence"][concept] = (traditional_conf + confidence) / 2
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid confidence level format: {context_analysis.get('confidence_level')}")

        enhanced_predictions = traditional_predictions.copy()
        enhanced_predictions["contextual_factors"] = {
            concept: self.interpretations["trend_explanations"].get(concept, {})
            for concept_data in all_top_concepts
            for concept in [concept_data["concept"]]
        }

        enhanced_predictions["confidence_scores"] = self.confidence_scores
        return enhanced_predictions
    
    def predict_research_trajectory_with_narrative(self, prediction_years=3):
        # First find traditional method to predict trajectories
        traditional_predictions = self.predict_research_trajectories(prediction_years)

        enhanced_predictions = {}

        # Process top trajectories
        trajectory_ids = list(traditional_predictions.keys())[:5]

        for trajectory_id in trajectory_ids:
            traj_data = traditional_predictions[trajectory_id]
            # Extract key information
            concepts = traj_data.get("concepts", [])
            status = traj_data.get("status", "unknown")
            recent_slope = traj_data.get("recent_slope", 0)

            # Prepare historical data string
            historical_years = sorted(traj_data.get("historical_data", {}).keys())
            historical_data_str = ", ".join([
                f"{year}: {traj_data['historical_data'][year]:.1f}" 
                for year in historical_years
            ])
            # Prepare predicted data string
            future_years = traj_data.get("future_years", [])
            predicted_values = traj_data.get("predicted_values", [])
            predicted_data_str = ", ".join([
                f"{year}: {value:.1f}" 
                for year, value in zip(future_years, predicted_values)
            ])
            prompt = f"""
            You are a scientific research analyst specializing in trajectory analysis and forecasting.
            
            Research Trajectory:
            - Concepts: {', '.join(concepts)}
            - Current Status: {status}
            - Recent Trend: {recent_slope:.2f}
            - Historical Data: {historical_data_str}
            - Predicted Values: {predicted_data_str}
            
            Create a narrative analysis of this research trajectory that explains:
            1. The overall narrative arc of this research area
            2. The key inflection points in its development
            3. The likely future developments based on the prediction
            4. Potential disruptive factors that could alter this trajectory
            5. How this trajectory relates to broader scientific or technological trends
            
            Provide your narrative analysis as a JSON object with fields for "narrative_arc", "inflection_points", 
            "future_developments", "disruptive_factors", and "broader_context".
            """
            llm_response = self._call_llm_api(prompt, temperature=0.4)
            narrative = {}
            if llm_response:
                try:
                    narrative = json.loads(llm_response)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse LLM response as JSON for narrative analysis")
                    # Try to extract JSON using regex
                    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                    match = re.search(json_pattern, llm_response)
                    if match:
                        try:
                            narrative = json.loads(match.group(0))
                        except:
                            self.logger.error("Failed to extract JSON from LLM response")

            self.interpretations["research_narratives"][trajectory_id] = narrative

            # Create enhanced predictions
            enhanced_prediction = traj_data.copy()
            enhanced_prediction["narrative"] = narrative
            enhanced_predictions[trajectory_id] = enhanced_prediction
    
        for trajectory_id in set(traditional_predictions.keys()) - set(enhanced_predictions.keys()):
            enhanced_predictions[trajectory_id] = traditional_predictions[trajectory_id]

        return enhanced_predictions

    def identify_influential_papers(self):
        """
        Identify influential papers based on citation metrics and concept relevance.
        
        Returns:
            List of influential papers with metadata
        """
        # Placeholder implementation - would need to be implemented based on data structure
        self.logger.info("Identifying influential papers")
        return []
    
    def generate_enhanced_insights(self):
        # First get traditional insights
        traditional_insights = self.generate_insights()  # Fixed: corrected method name
        # Collect all data from comprehensive analysis
        analysis_data = {
            "trending_concepts": self.identified_patterns.get("trending_concepts", []),
            "concept_clusters": self.identified_patterns.get("concept_clusters", []),
            "research_trajectories": self.identified_patterns.get("research_trajectories", []),
            "research_gaps": self.enhanced_patterns.get("research_gaps", []),
            "interdisciplinary_connections": self.enhanced_patterns.get("interdisciplinary_connections", [])
        }
        # Prepare data summaries for LLM
        trends_summary = "\n".join([
            f"- {t['concept']} (Growth: {t['growth_rate']:.2f})" 
            for t in analysis_data["trending_concepts"][:5]
        ])
    
        gaps_summary = ""
        if analysis_data["research_gaps"]:
            if isinstance(analysis_data["research_gaps"][0], dict):
                gaps_summary = "\n".join([
                f"- {g.get('description', 'Unknown gap')}" 
                for g in analysis_data["research_gaps"][:3]
            ])
            else:  # If they're strings
                gaps_summary = "\n".join([
                f"- {g}" for g in analysis_data["research_gaps"][:3]
            ])
       # print("this is gaps summary")
        #print(gaps_summary)
        connections_summary = ""
        if analysis_data["interdisciplinary_connections"] and len(analysis_data["interdisciplinary_connections"]) > 0:
        # Check if the first item is a dictionary or string
            if isinstance(analysis_data["interdisciplinary_connections"][0], dict):
                connections_summary = "\n".join([
                f"- {c.get('areas', 'Unknown connection')}: {c.get('description', '')}" 
                for c in analysis_data["interdisciplinary_connections"][:3]
            ])
            else:  # If they're strings
                connections_summary = "\n".join([
                f"- {c}" for c in analysis_data["interdisciplinary_connections"][:3]
            ])
        print("this is connection summary")
        print(connections_summary)

        prompt = f"""
        You are a scientific research strategist specializing in identifying high-value research directions.
        
        Based on the following analysis of research patterns:
        
        Top Trending Concepts:
        {trends_summary}
        
        Key Research Gaps:
        {gaps_summary}
        
        Interdisciplinary Connections:
        {connections_summary}
        
        Generate 3-5 high-level strategic insights that synthesize this information into actionable research strategies.
        For each insight:
        1. Provide a clear, concise title
        2. Explain the insight and its significance
        3. Describe the evidence supporting this insight
        4. Suggest specific research directions or approaches
        5. Assess potential impact and feasibility
        
        Focus on insights that would not be obvious from looking at individual patterns alone.
        Provide your insights as a JSON array of objects with fields for "title", "explanation", "supporting_evidence", 
        "research_directions", and "impact_assessment".
        """
        print(analysis_data)
        llm_response = self._call_llm_api(prompt, temperature=0.4, max_tokens=1500)
       # print(llm_response)
        enhanced_insights = []
    
        if llm_response:
            try:
                enhanced_insights = json.loads(llm_response)
                if isinstance(enhanced_insights, dict) and "insights" in enhanced_insights:
                    enhanced_insights = enhanced_insights["insights"]
            except (json.JSONDecodeError, TypeError):
                self.logger.error("Failed to parse LLM response as JSON for enhanced insights")
                # Try to extract JSON using regex
                json_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'
                match = re.search(json_pattern, llm_response)
                if match:
                    try:
                        enhanced_insights = json.loads(match.group(0))
                    except:
                        self.logger.error("Failed to extract JSON array from LLM response")
        combined_insights = []

        for insight in enhanced_insights:
            combined_insights.append({
                "insight_type": "strategic_direction",
                "title": insight.get("title", "Unknown insight"),
                "explanation": insight.get("explanation", ""),
                "supporting_evidence": insight.get("supporting_evidence", ""),
                "research_directions": insight.get("research_directions", []),
                "impact_assessment": insight.get("impact_assessment", "")
            })
        for insight in traditional_insights:
            if not any(i for i in combined_insights if i["insight_type"] == insight["insight_type"]):
                combined_insights.append(insight)

        return combined_insights
