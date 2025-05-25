class LLMImpactPredictor(BaseLLMExpert):
    """LLM-powered expert for predicting research impact"""
    
    def __init__(self, llm_interface: LLMInterface):
        super().__init__("LLM Impact Predictor", "Research Impact Assessment", llm_interface)
        
    def get_system_prompt(self) -> str:
        return """You are a renowned research impact analyst with expertise in predicting how scientific research will influence academia, society, and industry. Your capabilities include:

1. Assessing scientific impact potential based on novelty and rigor
2. Evaluating societal benefits and real-world applications
3. Predicting commercial and economic implications
4. Understanding research dissemination and adoption patterns
5. Identifying long-term breakthrough potential

You have studied the impact trajectories of thousands of research projects and can accurately predict which research directions will create the most value for science and society. You understand both immediate and long-term impact pathways."""

    def process(self, assessment: FeasibilityAssessment) -> ImpactPrediction:
        self.log_processing("FeasibilityAssessment", "ImpactPrediction")
        
        concept = assessment.source_concept
        
        prompt = f"""
RESEARCH IMPACT PREDICTION ANALYSIS

Research Title: {concept.title}
Hypothesis: {concept.hypothesis}
Innovation Rationale: {concept.innovation_rationale}
Expected Outcomes: {concept.expected_outcomes}
Feasibility: {assessment.overall_feasibility.value}
Success Probability: {assessment.success_probability}
Resource Requirements: {json.dumps(assessment.resource_analysis, indent=2)}

Predict the potential impact of this research across multiple dimensions if successfully executed.

Format your response as JSON:

{{
    "scientific_impact": "breakthrough|significant|moderate|incremental",
    "scientific_impact_reasoning": "Detailed analysis of potential scientific contributions",
    "societal_impact": "breakthrough|significant|moderate|incremental", 
    "societal_impact_reasoning": "How this research could benefit society",
    "commercial_potential": "breakthrough|significant|moderate|incremental",
    "commercial_reasoning": "Commercial applications and economic potential",
    "impact_timeline": "When significant impact is likely to be realized",
    "target_beneficiaries": [
        "Primary group who will benefit",
        "Secondary beneficiaries",
        "Long-term beneficiaries"
    ],
    "broader_implications": [
        "Implication for the research field",
        "Implications for related fields",
        "Societal and policy implications",
        "Technological implications"
    ],
    "potential_breakthroughs": [
        "Specific breakthrough that could emerge",
        "Revolutionary application potential",
        "Paradigm-shifting discoveries possible"
    ],
    "dissemination_strategy": "How research should be disseminated for maximum impact",
    "follow_up_research": "What follow-up research this could enable",
    "scaling_potential": "How the impact could scale beyond initial research",
    "risk_factors": "Factors that could limit impact realization",
    "impact_metrics": {{
        "citation_potential": "Expected academic citation impact",
        "application_adoption": "Likelihood of practical application",
        "field_influence": "Influence on research field direction",
        "societal_reach": "Breadth of societal impact"
    }}
}}

Consider both direct and indirect impact pathways. Think about how this research could catalyze other advances.
"""

        response = await self.llm.generate_response(prompt, self.get_system_prompt(), temperature=0.7)
        
        try:
            parsed_response = json.loads(response)
            
            # Convert string impacts to enums
            impact_map = {
                "breakthrough": ImpactLevel.BREAKTHROUGH,
                "significant": ImpactLevel.SIGNIFICANT,
                "moderate": ImpactLevel.MODERATE,
                "incremental": ImpactLevel.INCREMENTAL
            }
            
            return ImpactPrediction(
                prediction_id=str(uuid.uuid4()),
                source_assessment=assessment,
                scientific_impact=impact_map.get(parsed_response.get("scientific_impact", "moderate"), ImpactLevel.MODERATE),
                societal_impact=impact_map.get(parsed_response.get("societal_impact", "moderate"), ImpactLevel.MODERATE),
                commercial_potential=impact_map.get(parsed_response.get("commercial_potential", "moderate"), ImpactLevel.MODERATE),
                impact_timeline=parsed_response.get("impact_timeline", ""),
                target_beneficiaries=parsed_response.get("target_beneficiaries", []),
                broader_implications=parsed_response.get("broader_implications", []),
                potential_breakthroughs=parsed_response.get("potential_breakthroughs", []),
                impact_metrics=parsed_response.get("impact_metrics", {})
            )
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            raise ValueError("Invalid JSON response from LLM")
