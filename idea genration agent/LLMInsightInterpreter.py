class LLMInsightInterpreter:
    def __init__(self,llm_interface:LLMInterface):
        super().__init__("LLM Insight Interpreter", "Research Opportunity Identification",llm_interface)
    def get_system_prompt(self) -> str:
        return  """You are an expert research strategist specializing in converting research insights into structured research opportunities. Your role is to:
        1. Analyze research insights and extract actionable research opportunities
        2. Generate multiple high-quality research questions that address the insight
        3. Identify the most promising research approaches and methodologies
        4. Assess the novelty and urgency of the research opportunity
        5. Determine required expertise and interdisciplinary connections
        
        You have deep knowledge across multiple research domains including neuroscience, AI/ML, biology, psychology, medicine, and computer science. You excel at identifying cross-disciplinary opportunities and novel research directions.
        
        Your responses should be analytical, creative, and grounded in current research methodologies. Focus on generating opportunities that are both scientifically rigorous and potentially impactful."""
    def process(self,insight:ResearchInsight)->StructuredOpportunity:
        prompt = f"""
        RESEARCH INSIGHT ANALYSIS
        
        Category: {insight.category}
        Title: {insight.title}
        Description: {insight.description}
        Supporting Evidence: {', '.join(insight.supporting_evidence)}
        Domain: {insight.domain.value}
        Confidence Score: {insight.confidence_score}
        
        Please analyze this research insight and provide a structured opportunity assessment. Format your response as JSON with the following structure:
        
        {{
            "research_questions": [
                "Primary research question that addresses the core insight",
                "Secondary research question exploring a different angle",
                "Interdisciplinary research question connecting to other domains"
            ],
            "potential_approaches": [
                "Detailed methodology approach 1",
                "Alternative methodology approach 2", 
                "Novel/experimental approach 3",
                "Interdisciplinary approach 4"
            ],
            "required_expertise": [
                "Primary domain expertise",
                "Complementary expertise area",
                "Technical/methodological expertise"
            ],
            "relevant_domains": [
                "Primary domain and any additional relevant domains"
            ],
            "novelty_assessment": "Detailed assessment of what makes this opportunity novel and significant",
            "urgency_rationale": "Analysis of why this research is urgent and timely",
            "cross_disciplinary_potential": "Assessment of potential for cross-disciplinary collaboration",
            "resource_implications": "Initial thoughts on resource requirements"
        }}
        
        Focus on generating truly innovative and actionable research opportunities that could lead to significant advances in the field.
        """
        response = self.llm.generate_response(prompt,self.get_system_prompt(),temperature=0.8)
        try:
            parsed_response = json.loads(response)
            
            return StructuredOpportunity(
                opportunity_id=str(uuid.uuid4()),
                source_insight=insight,
                research_questions=parsed_response.get("research_questions", []),
                potential_approaches=parsed_response.get("potential_approaches", []),
                required_expertise=parsed_response.get("required_expertise", []),
                relevant_domains=[insight.domain],  # Could be enhanced to parse from response
                novelty_assessment=parsed_response.get("novelty_assessment", ""),
                urgency_rationale=parsed_response.get("urgency_rationale", ""),
                structured_metadata={
                    "cross_disciplinary_potential": parsed_response.get("cross_disciplinary_potential", ""),
                    "resource_implications": parsed_response.get("resource_implications", ""),
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            raise ValueError("Invalid JSON response from LLM")
        
