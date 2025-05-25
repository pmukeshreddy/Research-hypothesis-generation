class LLMFinal:
    def __init__(self):
        self.llm_inference = LLMInterface(GOOGLE,"AIzaSyBiGVSuaYDOgo7f5pB64l25OlQGnTc9tdY")
        self.insight_interpreter = LLMInsightInterpreter(self.llm_inference) # takes llm inisghts and converts to structured oppurtunities
        self.idea_genrator = LLMIdeaGenerator(self.llm_inference) # genrate and reaserach concept
        self.feasibility_assessor = LLMFeasibilityAssessor(self.llm_inference) # checks research feasabilty
        self.impact_predictor = LLMImpactPredictor(self.llm_inference)# this predicts research impact

        self.pipeline_id = str(uuid.uuid4())

    def genrate_idea(self,insight:ResearchInsight) ->GeneratedIdea:
        logger.info(f"Processing insight: {insight.title}")
        try:
            opportunity = self.insight_interpreter.process(insight)
            logger.info(f"Generated opportunity: {opportunity.opportunity_id}")
            concept = self.idea_genrator.process(opportunity)
            logger.info(f"Generated concept: {concept.concept_id}")
            feasibility = self.feasibility_assessor.process(concept)
            logger.info(f"Completed feasibility assessment: {feasibility.assessment_id}")
            impact = self.impact_predictor.process(feasibility)
            logger.info(f"Completed impact prediction: {impact.prediction_id}")

            final_score = self._calculate_final_score(feasibility,impact)
            recommendation = self._generate_recommendation(feasibility,impact,final_score)
            executive_summary = self._generate_executive_summary(opportunity,concept,feasibility,impact)

            idea = GeneratedIdea(idea_id=self.pipeline_id,
                                opportunity=opportunity,
                                concept=concept,
                                feasibility=feasibility,
                                impact=impact,
                                final_score=final_score,
                                recommendation=recommendation,
                                executive_summary=executive_summary)
            logger.info(f"Generated complete idea: {idea.idea_id} (score: {final_score:.2f})")
            return idea
            
        except Exception as e:
            logger.error(f"Error in LLM idea generation pipeline: {str(e)}")
            raise 
    def _calculate_final_score(self, feasibility: FeasibilityAssessment, impact: ImpactPrediction) -> float:
        """Calculate final composite score for the idea"""
        weights = {
            "feasibility": 0.25,
            "success_probability": 0.15,
            "scientific_impact": 0.25,
            "societal_impact": 0.20,
            "commercial_potential": 0.15
        }
        
        # Convert impact levels to scores
        impact_to_score = {
            ImpactLevel.BREAKTHROUGH: 1.0,
            ImpactLevel.SIGNIFICANT: 0.8,
            ImpactLevel.MODERATE: 0.6,
            ImpactLevel.INCREMENTAL: 0.4
        }
        
        feasibility_to_score = {
            FeasibilityLevel.HIGH: 0.9,
            FeasibilityLevel.MEDIUM: 0.7,
            FeasibilityLevel.LOW: 0.4,
            FeasibilityLevel.UNCERTAIN: 0.2
        }
        
        scores = {
            "feasibility": feasibility_to_score[feasibility.overall_feasibility],
            "success_probability": feasibility.success_probability,
            "scientific_impact": impact_to_score[impact.scientific_impact],
            "societal_impact": impact_to_score[impact.societal_impact],
            "commercial_potential": impact_to_score[impact.commercial_potential]
        }
        
        return sum(weights[key] * scores[key] for key in weights)
    def _generate_recommendation(self, feasibility: FeasibilityAssessment, impact: ImpactPrediction, score: float) -> str:
        """Generate LLM-powered recommendation"""
        prompt = f"""
Based on the research feasibility assessment and impact prediction, generate a concise recommendation for this research proposal.

Feasibility: {feasibility.overall_feasibility.value}
Success Probability: {feasibility.success_probability}
Scientific Impact: {impact.scientific_impact.value}
Societal Impact: {impact.societal_impact.value}
Commercial Potential: {impact.commercial_potential.value}
Composite Score: {score:.2f}

Provide a clear, actionable recommendation in 2-3 sentences that captures the essence of whether this research should be pursued and under what conditions.
"""
        
        return self.llm_interface.generate_response(
            prompt, 
            "You are a research advisory expert providing concise, actionable recommendations.",
            temperature=0.5,
            max_tokens=200
        )
    def _generate_executive_summary(self, opportunity: StructuredOpportunity, concept: ResearchConcept, 
                                        feasibility: FeasibilityAssessment, impact: ImpactPrediction) -> str:
        """Generate LLM-powered executive summary"""
        prompt = f"""
Create a compelling executive summary for this research proposal that captures the key elements:

Research Title: {concept.title}
Core Hypothesis: {concept.hypothesis}
Innovation: {concept.innovation_rationale}
Feasibility: {feasibility.overall_feasibility.value}
Scientific Impact: {impact.scientific_impact.value}
Societal Impact: {impact.societal_impact.value}
Timeline: {impact.impact_timeline}

Write a 3-4 sentence executive summary that would compel a research committee or funding agency to seriously consider this proposal. Focus on the unique value proposition and potential impact.
"""
        
        return await self.llm_interface.generate_response(
            prompt,
            "You are an expert grant writer creating compelling research summaries.",
            temperature=0.6,
            max_tokens=300
        )
