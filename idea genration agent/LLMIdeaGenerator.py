class LLMIdeaGenerator:
    def __init__(self,llm_interface:LLMInterface):
        super().__init__("LLM Idea Generator", "Research Concept Creation", llm_interface)
    def get_system_prompt(self) ->str:
                return """You are a brilliant research innovator with expertise across multiple scientific domains. Your specialty is generating creative, novel, and feasible research concepts that push the boundaries of current knowledge.

Your capabilities include:
1. Synthesizing insights from multiple research areas into innovative concepts
2. Designing rigorous experimental methodologies
3. Identifying breakthrough potential in research ideas
4. Balancing creativity with scientific rigor
5. Anticipating technical requirements and challenges

You think like the most innovative researchers in history - combining deep domain knowledge with creative thinking to generate ideas that could lead to significant scientific breakthroughs. Your concepts should be ambitious yet grounded in solid scientific principles."""

    def process(self,opportunity:StructuredOpportunity) ->ResearchConcept:
        self.log_processing("StructuredOpportunity", "ResearchConcept")
        prompt = f"""
RESEARCH OPPORTUNITY ANALYSIS

Source Insight: {opportunity.source_insight.title}
Research Questions: {opportunity.research_questions}
Potential Approaches: {opportunity.potential_approaches}
Required Expertise: {opportunity.required_expertise}
Novelty Assessment: {opportunity.novelty_assessment}
Urgency Rationale: {opportunity.urgency_rationale}

Based on this structured opportunity, generate an innovative research concept that could lead to significant scientific impact. Be creative and ambitious while maintaining scientific rigor.

Format your response as JSON:

{{
    "title": "Compelling title for the research concept",
    "hypothesis": "Clear, testable hypothesis that addresses the research opportunity",
    "methodology": {{
        "overall_approach": "High-level methodological framework",
        "experimental_design": "Detailed experimental design approach",
        "data_collection": "Data collection strategy and methods", 
        "analysis_plan": "Data analysis and interpretation approach",
        "validation_strategy": "How results will be validated and verified",
        "timeline_phases": ["Phase 1 description", "Phase 2 description", "Phase 3 description"]
    }},
    "expected_outcomes": [
        "Primary expected outcome",
        "Secondary expected outcomes",
        "Potential breakthrough discoveries",
        "Publications and dissemination strategy"
    ],
    "innovation_rationale": "Detailed explanation of what makes this concept innovative and why it could lead to breakthroughs",
    "technical_requirements": [
        "Specific equipment/technology needed",
        "Computational requirements", 
        "Specialized facilities or resources",
        "Key technical capabilities required"
    ],
    "creative_elements": [
        "Novel aspects of the approach",
        "Creative methodological innovations",
        "Unique interdisciplinary combinations",
        "Breakthrough potential elements"
    ],
    "risk_mitigation": "How key technical and scientific risks will be addressed",
    "scalability_potential": "How the concept could be scaled or extended"
}}

Focus on generating a concept that is both scientifically rigorous and creatively ambitious. Think about what could truly advance the field.
"""
        response = self.llm.genrate_response(prompt,self.get_system_prompt(),temperature=0.9)
        try:
            parsed_response = json.loads(response)
            return ResearchConcept(concept_id=str(uuid.uuid4()),#uuid4 means randomly genrated ones
                                  source_opportunity=opportunity,
                                  title=parsed_response.get("title",""),
                                  hypothesis=parsed_response.get("hypothesis",""),
                                  methodology=parsed_response.get("parsed_response",""),
                                expected_outcomes=parsed_response.get("expected_outcomes", []),
                                innovation_rationale=parsed_response.get("innovation_rationale", ""),
                                technical_requirements=parsed_response.get("technical_requirements", []),
                                creative_elements=parsed_response.get("creative_elements", []),
                                concept_details={
                    "risk_mitigation": parsed_response.get("risk_mitigation", ""),
                    "scalability_potential": parsed_response.get("scalability_potential", ""),
                    "generation_timestamp": datetime.now().isoformat()
                }
                                            ) 
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            raise ValueError("Invalid JSON response from LLM")
