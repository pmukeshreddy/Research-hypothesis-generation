class LLMFeasibilityAssessor:
    def __init__(self,llm_interface:LLMInterface):
        super().__init__("LLM Feasibility Assessor", "Research Viability Analysis", llm_interface)
    def get_system_prompt(self)->str:
                return """You are an expert research project manager and feasibility analyst with decades of experience evaluating research proposals across multiple domains. Your expertise includes:

1. Realistic resource estimation and project planning
2. Risk identification and mitigation strategy development  
3. Timeline estimation based on research complexity
4. Technical feasibility assessment
5. Budget and cost analysis for research projects
6. Success probability evaluation based on similar past projects

You have managed hundreds of research projects and can accurately assess what works in practice vs. theory. You provide honest, realistic assessments while identifying ways to maximize success probability. Your analysis helps researchers make informed decisions about project viability."""
    def process(self,concept:ResearchConcept)->FeasibilityAssessment:
        prompt = f"""
RESEARCH CONCEPT FEASIBILITY ANALYSIS

Title: {concept.title}
Hypothesis: {concept.hypothesis}
Methodology: {json.dumps(concept.methodology, indent=2)}
Technical Requirements: {concept.technical_requirements}
Innovation Rationale: {concept.innovation_rationale}
Expected Outcomes: {concept.expected_outcomes}

Conduct a comprehensive feasibility analysis of this research concept. Consider all practical aspects of executing this research.

Format your response as JSON:

{{
    "overall_feasibility": "high|medium|low|uncertain",
    "feasibility_reasoning": "Detailed reasoning for the overall assessment",
    "resource_analysis": {{
        "personnel_requirements": "Detailed staffing needs and expertise",
        "equipment_costs": "Equipment and infrastructure requirements",
        "operational_costs": "Ongoing operational expenses",
        "facility_requirements": "Lab space, specialized facilities needed"
    }},
    "timeline_breakdown": {{
        "preparation_phase": "Setup and preparation timeline",
        "execution_phase": "Main research execution timeline", 
        "analysis_phase": "Data analysis and interpretation timeline",
        "publication_phase": "Writing and publication timeline",
        "total_duration": "Overall project timeline"
    }},
    "risk_analysis": [
        {{
            "risk_type": "technical|resource|timeline|regulatory|other",
            "description": "Detailed risk description", 
            "probability": "high|medium|low",
            "impact": "high|medium|low",
            "mitigation": "Specific mitigation strategy"
        }}
    ],
    "success_probability": 0.75,
    "success_probability_reasoning": "Detailed reasoning for success probability estimate",
    "bottlenecks": [
        "Key constraint or bottleneck 1",
        "Key constraint or bottleneck 2"
    ],
    "mitigation_strategies": [
        "Strategic approach to address major risks",
        "Fallback plans and alternatives",
        "Ways to improve success probability"
    ],
    "cost_breakdown": {{
        "personnel_costs": "Estimated personnel costs",
        "equipment_costs": "Equipment and technology costs",
        "operational_costs": "Ongoing operational costs", 
        "total_estimated_cost": "Total project cost estimate"
    }},
    "optimization_recommendations": "How to improve feasibility and success probability"
}}

Be realistic but constructive in your assessment. Identify real challenges while suggesting practical solutions.
"""
        response = self.llm.genrate_response(prompt,self.get_system_prompt(),temperature=0.6)
        try:
            parsed_response = json.loads(response)
            
            # Convert string feasibility to enum
            feasibility_map = {
                "high": FeasibilityLevel.HIGH,
                "medium": FeasibilityLevel.MEDIUM,
                "low": FeasibilityLevel.LOW,
                "uncertain": FeasibilityLevel.UNCERTAIN
            }
            
            return FeasibilityAssessment(
                assessment_id=str(uuid.uuid4()),
                source_concept=concept,
                overall_feasibility=feasibility_map.get(parsed_response.get("overall_feasibility", "medium"), FeasibilityLevel.MEDIUM),
                resource_analysis=parsed_response.get("resource_analysis", {}),
                timeline_breakdown=parsed_response.get("timeline_breakdown", {}),
                risk_analysis=parsed_response.get("risk_analysis", []),
                success_probability=parsed_response.get("success_probability", 0.5),
                bottlenecks=parsed_response.get("bottlenecks", []),
                mitigation_strategies=parsed_response.get("mitigation_strategies", []),
                cost_breakdown=parsed_response.get("cost_breakdown", {})
            )
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            raise ValueError("Invalid JSON response from LLM")
