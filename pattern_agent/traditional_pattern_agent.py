from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime
from pmdarima import auto_arima
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


class PatternRecognitionAgent:
    def __init__(self,concept_graph):
        # Initialize necessary attributes
        self.concept_graph = concept_graph
        self.identified_patterns = {
            "trending_concepts": [],
            "concept_clusters": [],
            "research_trajectories": []
        }
        self.concept_growth_predictions = {}
        self.trajectory_predictions = {}
    
    def identify_trending_concepts(self, time_window=3, min_frequency=2):
        """
        Identify trending concepts based on frequency growth.
        Alternatives: Kleinberg's burst-detection algorithm, Dynamic topic modeling,
        Graph-based centrality with temporal weighting
        """
        print("Identifying traditional trending concepts")
        print("Graph nodes:", len(self.concept_graph.graph.nodes()))
        if not self.concept_graph:
            return []
            
        freq_by_year = self.concept_graph.get_concept_frequency_by_year()
        # Get range of years
        all_years = sorted(freq_by_year.keys())
      #  print("manus")
       # print(len(all_years))
        if len(all_years) < 2:
            return []
            
        # Check if we have enough years for window
        if len(all_years) < time_window:
            time_window = len(all_years)
        
        recent_years = all_years[-time_window:]
        earlier_years = all_years[:-time_window]
        if not earlier_years:
            earlier_years = [recent_years[0]]
            recent_years = recent_years[1:]
        if not recent_years:
            return []
        
        earlier_avg = defaultdict(float) 
        recent_avg = defaultdict(float) 
       # print("yuyu")
        #print(freq_by_year)
        all_concepts = set()
        for year_data in freq_by_year.values():
            all_concepts.update(year_data.keys())

        # Calculate averages
        for concept in all_concepts:
            # Earlier years
            earlier_sum = sum(freq_by_year[year].get(concept, 0) for year in earlier_years)
            earlier_avg[concept] = earlier_sum / len(earlier_years) if earlier_years else 0
            # Recent years - FIXED: using recent_years consistently
            recent_sum = sum(freq_by_year[year].get(concept, 0) for year in recent_years)
            recent_avg[concept] = recent_sum / len(recent_years) if recent_years else 0

        # Calculate growth and identify trending concepts
        trending = []

        for concept in all_concepts:
            # Skip concepts with few connections
            if recent_avg[concept] < min_frequency:
                continue
            # Calculate growth
            growth = 0
            if earlier_avg[concept] > 0:
                growth = (recent_avg[concept] - earlier_avg[concept]) / earlier_avg[concept]
            elif recent_avg[concept] > 0:
                growth = float("inf")
            # Consider it if growth is positive
            if growth > 0:
                trending.append({
                    "concept": concept,
                    "growth_rate": growth if growth != float("inf") else 999,
                    "recent_freq": recent_avg[concept],
                    "earlier_freq": earlier_avg[concept]
                })
        
        self.identified_patterns["trending_concepts"] = trending
        return trending

    def identify_concept_cluster(self, min_samples=3, eps=0.5):
        """
        Identify concept clusters using similarity matrix and DBSCAN clustering.
        Alternatives: community detection, embedding + HDBSCAN
        """
        print("Identifying concept clusters...")
        if not self.concept_graph:
            print("ERROR: No concept graph found!")
            return []
        if not self.concept_graph or not hasattr(self.concept_graph, "concept_mentions"):
            print("yeah reason 1")
            return []
            
        concepts = list(self.concept_graph.concept_mentions.keys())
        n_concepts = len(concepts)

        if n_concepts < 3:
            print("print it's reason 2")
            return []
        
        concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
        co_occurrence = np.zeros((n_concepts, n_concepts))        

        for paper_id in self.concept_graph.nodes_by_type.get("paper", []):
            paper_concepts = []

            # Get all the concepts mentioned in the paper
            for concept_id in self.concept_graph.nodes_by_type.get("concept", []):
                if self.concept_graph.graph.has_edge(paper_id, concept_id):
                    concept_name = self.concept_graph.graph.nodes[concept_id].get("name")
                    if concept_name in concept_to_idx:
                        paper_concepts.append(concept_name)
                        
            # Update co-occurrence matrix
            for i, concept1 in enumerate(paper_concepts):
                idx1 = concept_to_idx[concept1]
                for concept2 in paper_concepts[i+1:]:
                    idx2 = concept_to_idx[concept2]
                    co_occurrence[idx1, idx2] += 1
                    co_occurrence[idx2, idx1] += 1

        # Normalize to get similarity matrix
        concept_similarity = np.zeros((n_concepts, n_concepts))
        for i in range(n_concepts):
            for j in range(n_concepts):
                if i == j:
                    concept_similarity[i, j] = 1.0
                else:
                    if co_occurrence[i, i] + co_occurrence[j, j] - co_occurrence[i, j] > 0:
                        concept_similarity[i, j] = co_occurrence[i, j] / (co_occurrence[i, i] + co_occurrence[j, j] - co_occurrence[i, j])
        
        # Apply clustering
        clustering = DBSCAN(eps=0.9, min_samples=1, metric='precomputed')
        distance_matrix = 1 - concept_similarity
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Organize clusters
        clusters = defaultdict(list)
        # FIXED: Using enumerate instead of range
        for i, label in enumerate(cluster_labels):
            if label >= 0:
                clusters[label].append(concepts[i])
        
        # Convert to list of clusters
        cluster_list = [{"cluster_id": i, "concepts": c} for i, c in clusters.items()]
        self.identified_patterns["concept_clusters"] = cluster_list

      #  print("checking for concepts list")
     #   print(cluster_list)
        return cluster_list


        
            
    def predict_concept_growth(self, prediction_years=3):
        """
        Predict future growth of concepts using time series forecasting.
        """
        print("Predicting concept growth...")

        if not self.concept_graph or not hasattr(self.concept_graph, "get_concept_frequency_by_year"):
            return {}

        freq_by_year = self.concept_graph.get_concept_frequency_by_year()
        all_years_sorted = sorted(freq_by_year.keys())

        if len(all_years_sorted) < 5:
            return {}

        predictions = {}

        for concept in self.concept_graph.concept_mentions:
            # Get yearly data
            yearly_data = []

            for year in all_years_sorted:
                yearly_data.append((year, freq_by_year[year].get(concept, 0)))
            
            # Skip concepts with few data points
            if sum(y[1] for y in yearly_data) < 10:
                continue

            df = pd.DataFrame(yearly_data, columns=["year", "frequency"])

            model_choice = self._select_best_forecast_model(df)

            try:
                if model_choice == "prophet":
                    prophet_df = df.rename(columns={"year": "ds", "frequency": "y"})
                    model = Prophet(
                        yearly_seasonality=True, 
                        growth='linear',
                        interval_width=0.95
                    )
                    model.fit(prophet_df)
                    # Create future dataframe
                    future = pd.DataFrame(
                        {'ds': pd.date_range(start=str(all_years_sorted[-1]), 
                                            periods=prediction_years+1, 
                                            freq='Y')}
                    )

                    forecast = model.predict(future)

                    future_years = forecast["ds"].dt.year.tolist()[1:]
                    predicted_values = forecast["yhat"].tolist()[1:]

                    # Store additional metrics
                    trend_components = forecast[["ds", "trend"]].iloc[1:].values
                    seasonality = True if "yearly" in forecast.columns else False
                
                elif model_choice == "arima":
                    arima_model = auto_arima(df["frequency"],
                                            start_p=0, start_q=0,
                                            max_p=3, max_q=3,
                                            seasonal=False,
                                            d=None, trace=False,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True)
                    
                    # Forecast future values 
                    forecast, conf_int = arima_model.predict(n_periods=prediction_years, return_conf_int=True)
                    # Store predictions
                    future_years = [all_years_sorted[-1]+i for i in range(1, prediction_years+1)]
                    predicted_values = [max(0, val) for val in forecast]
                    # Store ARIMA order
                    order = arima_model.order
                else:
                    X = df["year"].values.reshape(-1, 1)
                    y = df["frequency"].values

                    # Fit Ridge regression
                    model = Ridge(alpha=0.1)
                    model.fit(X, y)

                    # Predict the future values
                    future_years = [all_years_sorted[-1] + i for i in range(1, prediction_years+1)]
                    X_future = np.array(future_years).reshape(-1, 1)
                    predicted_values = [max(0, val) for val in model.predict(X_future)]
                
                # Calculate growth metrics
                current_value = freq_by_year[all_years_sorted[-1]].get(concept, 0)
                final_predicted = predicted_values[-1]

                growth_rate = 0
                if current_value > 0:
                    growth_rate = (final_predicted-current_value) / current_value
                
                # Store prediction with model metadata
                predictions[concept] = {
                    "historical_data": yearly_data,
                    "future_years": future_years,
                    "predicted_values": predicted_values,
                    "growth_rate": growth_rate,
                    "current_value": current_value,
                    "final_prediction": final_predicted,
                    "forecast_model": model_choice,
                    "confidence": self._calculate_forecast_confidence(model_choice, df, yearly_data)
                }

                # Add model-specific data
                if model_choice == "prophet":
                    predictions[concept]["trend_components"] = trend_components
                    predictions[concept]["has_seasonality"] = seasonality
                elif model_choice == "arima":
                    predictions[concept]["arima_order"] = order
            except Exception as e:
                print(f"Error forecasting for concept '{concept}': {str(e)}")
                continue

        # Sort predictions based on predicted growth rate
        sorted_concepts = sorted(predictions.keys(),
                                key=lambda c: predictions[c]["growth_rate"],
                                reverse=True)
        
        # Organize predictions
        concept_growth_predictions = {
            "prediction_years": prediction_years,
            "last_known_year": datetime.now().year if not self.concept_graph else all_years_sorted[-1] if all_years_sorted else datetime.now().year,
            "fast_growing": [],
            "stable": [],
            "declining": [],
            "all_predictions": predictions
        }

        for concept in sorted_concepts:
            growth_rate = predictions[concept]["growth_rate"]
            if growth_rate > 0.2:  # Fast growing
                concept_growth_predictions["fast_growing"].append({
                    "concept": concept,
                    "growth_rate": growth_rate,
                    "current_value": predictions[concept]["current_value"],
                    "final_predicted": predictions[concept]["final_prediction"],
                    "forecast_model": predictions[concept]["forecast_model"]
                })
            elif growth_rate >= -0.1:  # Stable
                concept_growth_predictions["stable"].append({
                    "concept": concept,
                    "growth_rate": growth_rate,
                    "current_value": predictions[concept]["current_value"],
                    "final_predicted": predictions[concept]["final_prediction"],
                    "forecast_model": predictions[concept]["forecast_model"]
                })
            else:  # Declining
                concept_growth_predictions["declining"].append({
                    "concept": concept,
                    "growth_rate": growth_rate,
                    "current_value": predictions[concept]["current_value"],
                    "final_predicted": predictions[concept]["final_prediction"],
                    "forecast_model": predictions[concept]["forecast_model"]
                })
        
        self.concept_growth_predictions = concept_growth_predictions
        return concept_growth_predictions

    def _select_best_forecast_model(self, df):
        """
        Determine the best forecasting model based on data characteristics.
        
        Args:
            df: DataFrame with year and frequency columns
            
        Returns:
            String indicating the best model to use
        """
        # Check data size
        if len(df) < 8:
            return "ridge"  # Use simpler model for limited data
            
        # Check for seasonality or patterns
        try:
            # Set the time series index
            ts = df.set_index('year')['frequency']
            
            # Check stationarity
            adf_result = adfuller(ts)
            
            # If p-value > 0.05, the series is non-stationary
            is_stationary = adf_result[1] <= 0.05
            
            # Check if there's enough data for seasonal decomposition
            if len(ts) >= 4:
                # Try to detect seasonality
                try:
                    result = seasonal_decompose(ts, model='additive', period=2)
                    seasonal_strength = np.std(result.seasonal) / np.std(ts)
                    has_seasonality = seasonal_strength > 0.1
                except:
                    has_seasonality = False
            else:
                has_seasonality = False
                
            # Decision logic
            if has_seasonality or not is_stationary:
                return "prophet"  # Better for non-stationary data with seasonality
            else:
                return "arima"  # Good for stationary data
                
        except Exception:
            # Default to Prophet for most cases
            return "prophet"
        
    def _calculate_forecast_confidence(self, model_type, df, yearly_data):
        """
        Calculate a confidence score for the forecast based on data quality and model fit.
        
        Args:
            model_type: String indicating which model was used
            df: DataFrame with the historical data
            yearly_data: Original yearly data points
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Adjust based on data quantity
        data_points = len(df)
        if data_points >= 10:
            confidence += 0.2
        elif data_points >= 7:
            confidence += 0.1
        else:
            confidence -= 0.1
            
        # Adjust based on data consistency
        frequencies = df['frequency'].values
        if len(frequencies) > 3:
            # Calculate coefficient of variation
            cv = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else float('inf')
            
            # Very stable data gets bonus
            if cv < 0.3:
                confidence += 0.1
            # Very erratic data gets penalty
            elif cv > 1.0:
                confidence -= 0.2
        
        # Model-specific adjustments
        if model_type == 'prophet':
            confidence += 0.05  # Slight bonus for Prophet's ability to handle trends
        elif model_type == 'ridge':
            confidence -= 0.1  # Penalty for falling back to simpler model
            
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))
    
    def identify_research_trajectories(self):
        """
        Identify research trajectories based on concept clusters over time.
        """
        print("Identifying research trajectories...")
        # First make sure we have concept clusters
        if not self.identified_patterns["concept_clusters"]:
            self.identify_concept_cluster()

        # Use concept graph if available
        if not self.concept_graph or not hasattr(self.concept_graph, "nodes_by_type"):
            return []

        trajectories = []

        papers_by_years = defaultdict(list)
        for paper_id in self.concept_graph.nodes_by_type.get("paper", []):
            paper_data = self.concept_graph.graph.nodes[paper_id]
            year = paper_data.get("year")
            if year:
                papers_by_years[year].append((paper_id, paper_data))
        
        for cluster in self.identified_patterns["concept_clusters"]:
            cluster_concepts = set(cluster["concepts"])

            # Track concept mention by year
            yearly_mentions = defaultdict(int)
            yearly_papers = defaultdict(list)

            # Find papers that mention concepts in cluster
            for year, papers in papers_by_years.items():
                for paper_id, paper_data in papers:
                    paper_concepts = set()
                    # Get concepts mentioned in that paper
                    for concept_id in self.concept_graph.nodes_by_type.get("concept", []):
                        if self.concept_graph.graph.has_edge(paper_id, concept_id):
                            concept_name = self.concept_graph.graph.nodes[concept_id].get("name")
                            paper_concepts.add(concept_name)

                    overlap = cluster_concepts.intersection(paper_concepts)
                    if overlap:
                        overlap_ratio = len(overlap) / len(cluster_concepts)
                        yearly_mentions[year] += len(overlap)
                   #     print(f"i am printing yearly mention :{yearly_mentions[year]}")
                        yearly_papers[year].append({
                            "paper_id": paper_id,
                            "title": paper_data.get("title", ""),
                            "overlap_concepts": list(overlap),
                            "overlap_ratio": overlap_ratio
                        })

            # Skip this cluster if no mentions
            if not yearly_mentions:
                continue
                
            # Sort papers by overlap ratio in each year  
            for year in yearly_papers:
                yearly_papers[year].sort(key=lambda x: x["overlap_ratio"], reverse=True)

            top_papers = []
            for year in sorted(yearly_papers.keys()):
                if yearly_papers[year]:
                    top_papers.append(yearly_papers[year][0])

            trajectory = {
                "cluster_id": cluster["cluster_id"],
                "concepts": cluster["concepts"],
                "yearly_mentions": dict(yearly_mentions),
                "top_papers": top_papers,
                "start_year": min(yearly_mentions.keys()),
                "end_year": max(yearly_mentions.keys()),
                "duration": max(yearly_mentions.keys()) - min(yearly_mentions.keys()) + 1
            }
                
            trajectories.append(trajectory)
            
        trajectories.sort(key=lambda x: x["duration"], reverse=True)
        self.identified_patterns["research_trajectories"] = trajectories
       # print("to check trajecotyr")
       # print(trajectories)
        return trajectories

    def predict_research_trajectories(self, prediction_years=3):
        """
        Predict future research trajectory trends.
        """
        print("Predicting research trajectories...")
        # First make sure we have research trajectories
        if not self.identified_patterns["research_trajectories"]:
            self.identify_research_trajectories()
        
        # Make sure we have concept growth predictions
        if not self.concept_growth_predictions:
            self.predict_concept_growth(prediction_years)
        
        trajectory_predictions = {}
        

        # Process each trajectory
        for trajectory in self.identified_patterns["research_trajectories"]:
            trajectory_id = trajectory["cluster_id"]
            concepts = trajectory["concepts"]
            yearly_mentions = trajectory.get("yearly_mentions", {})


            yearly_mentions_numeric = {}
            for year, count in yearly_mentions.items():
                year_int = int(str(year).strip()) if isinstance(year, str) else year
                yearly_mentions_numeric[year_int] = count
            
            # Skip trajectories with insufficient data
            if len(yearly_mentions) < 5:
                continue
                
            # Get years range for this trajectory
            #trajectory_years = sorted([int(year) if isinstance(year, str) else year for year in yearly_mentions.keys()])
            

            # Prepare data for prediction
            #time_series_data = np.array([yearly_mentions.get(str(year) if isinstance(year, int) else year, 0) for year in trajectory_years])
          #  print("this is yearly metions")
          #  print(yearly_mentions)
            trajectory_years = sorted(yearly_mentions_numeric.keys())

            time_series_data = np.array([yearly_mentions_numeric[year] for year in trajectory_years])

           # print("Chronologically ordered years:", trajectory_years)
          #  print("Corresponding time series data:", time_series_data)





            has_nonzero_data = any(val > 0 for val in time_series_data)
           # print("this is time series data")
          #  print(time_series_data)
            print(has_nonzero_data)

            # Try different time series models and use the best one
            try:
                best_aic = float("inf")
                best_model = None
                best_order = None

                # Try with different parameters
                p_values = range(0, 3)
                d_values = range(0, 3)
                q_values = range(0, 3)

                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            try:
                                model = ARIMA(time_series_data, order=(p, d, q))
                                model_fit = model.fit()
                                
                                if model_fit.aic < best_aic:
                                    best_aic = model_fit.aic
                                    best_model = model_fit
                                    best_order = (p, d, q)
                            except:
                                continue
                                
                # If ARIMA model fitting succeeded, use it
                if best_model is not None:
                    forecast = best_model.forecast(steps=prediction_years)
                    predicted_values = [max(0, value) for value in forecast]  # Ensure non-negative
                    model_type = f"ARIMA{best_order}"
                else:
                    # Fall back to Exponential Smoothing
                    model = ExponentialSmoothing(
                        time_series_data,
                        trend='add',  # Additive trend component
                        seasonal=None,  # No seasonal component
                        seasonal_periods=None
                    )
                    model_fit = model.fit()
                    forecast = model_fit.forecast(prediction_years)
                    predicted_values = [max(0, value) for value in forecast]  # Ensure non-negative
                    model_type = "Exponential Smoothing"

            except Exception as e:
                # Fall back to simple exponential smoothing if all else fails
                alpha = 0.3  # Smoothing factor
                level = time_series_data[0]
                predicted_values = []
            
                # Apply simple exponential smoothing
                for i in range(len(time_series_data)):
                    level = alpha * time_series_data[i] + (1 - alpha) * level
                
                # Forecast future values
                for _ in range(prediction_years):
                    predicted_values.append(max(0, level))  # Ensure non-negative
                
                model_type = "Simple Exponential Smoothing"

            future_years = [trajectory_years[-1] + i for i in range(1, prediction_years + 1)]

            # Calculate trajectory momentum (using last 3 years)
            recent_slope = 0
            if len(trajectory_years) >= 3:
                recent_X = np.array(trajectory_years[-3:]).reshape(-1, 1)
                recent_Y = np.array([yearly_mentions.get(str(year) if isinstance(year, int) else year, 0) for year in trajectory_years[-3:]])

                recent_X_mean = np.mean(recent_X)
                recent_Y_mean = np.mean(recent_Y)

                recent_numerator = np.sum((recent_X - recent_X_mean) * (recent_Y - recent_Y_mean))
                recent_denominator = np.sum((recent_X - recent_X_mean) ** 2)

                if recent_denominator != 0:
                    recent_slope = recent_numerator / recent_denominator

            # Get concept growth trends for concepts in trajectory
            concept_trend = []
            avg_concept_growth = 0
            growth_concepts = 0

            for concept in concepts:
                if self.concept_growth_predictions.get("all_predictions") and concept in self.concept_growth_predictions["all_predictions"]:
                    growth_rate = self.concept_growth_predictions["all_predictions"][concept]["growth_rate"]
                    concept_trend.append({
                        "concept": concept,
                        "growth_rate": growth_rate
                    })
                    avg_concept_growth += growth_rate
                    growth_concepts += 1

            if growth_concepts > 0:
                avg_concept_growth /= growth_concepts
            
            # Determine trajectory status
            status = "unknown"
            growth_trend = np.mean(predicted_values) - time_series_data[-1]

            if recent_slope > 0.1 and avg_concept_growth > 0.1 and growth_trend > 0:
                status = 'emerging'
            elif recent_slope > 0 and avg_concept_growth > 0 and growth_trend > 0:
                status = "growing"
            elif recent_slope < -0.1 and avg_concept_growth < -0.1 and growth_trend < 0.1:
                status = "declining"
            elif abs(recent_slope) <= 0.1 and abs(avg_concept_growth) <= 0.1 and abs(growth_trend) <= 0.1:
                status = "stable" 
            
            # Store predictions
            trajectory_predictions[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "concepts": concepts,
                "historical_data": {str(year): yearly_mentions.get(str(year) if isinstance(year, int) else year, 0) for year in trajectory_years},
                "future_years": future_years,
                "predicted_values": predicted_values,
                "model_type": model_type,
                "recent_slope": recent_slope,
                "avg_concept_growth": avg_concept_growth,
                "concept_trends": concept_trend,
                "status": status
            }
            
      #  self.trajectory_predictions = trajectory_predictions
       # print(trajectory_predictions)
        return trajectory_predictions
        
    def generate_insights(self):
        """
        Generate insights based on identified patterns and predictions.
        """
        insights = []

        # Ensure we have all necessary patterns and predictions
        if not self.identified_patterns.get("trending_concepts"):
            self.identify_trending_concepts()
        if not self.identified_patterns.get("concept_clusters"):
            self.identify_concept_cluster()
        if not self.identified_patterns.get("research_trajectories"):
            self.identify_research_trajectories()
        if not self.concept_growth_predictions:
            self.predict_concept_growth()
        if not self.trajectory_predictions:
            self.predict_research_trajectories()
        
        # Insight 1: Emerging research areas
        if self.identified_patterns["trending_concepts"] and self.identified_patterns["research_trajectories"]:
            emerging_areas = []

            for trajectory in self.identified_patterns["research_trajectories"]:
                trajectory_id = trajectory["cluster_id"]

                # Check if trajectory is in predictions
                if (trajectory_id in self.trajectory_predictions and 
                    self.trajectory_predictions[trajectory_id]["status"] in ["emerging", "growing"]):
                    # Get trending concepts in trajectory
                    trending_in_trajectory = []
                    trajectory_concepts = set(trajectory["concepts"])

                    for trend in self.identified_patterns["trending_concepts"]:
                        if trend["concept"] in trajectory_concepts:
                            trending_in_trajectory.append(trend["concept"])
                    
                    if trending_in_trajectory:
                        emerging_areas.append({
                            "trajectory_id": trajectory_id,
                            "concepts": trajectory["concepts"],
                            "trending_concepts": trending_in_trajectory,
                            "status": self.trajectory_predictions[trajectory_id]["status"],
                            "growth_rate": self.trajectory_predictions[trajectory_id]["avg_concept_growth"]
                        })
            
            if emerging_areas:
                insights.append({
                    "insight_type": "emerging_research_areas",
                    "description": "Identified emerging research areas based on trajectory analysis",
                    "areas": emerging_areas
                })

        # Insight 2: Research gaps
        if self.concept_growth_predictions and self.identified_patterns["concept_clusters"]:
            potential_gaps = []
            # Check each cluster for uneven growth patterns
            for cluster in self.identified_patterns["concept_clusters"]:
                cluster_concepts = cluster["concepts"]

                if len(cluster_concepts) < 3:
                    continue
                
                growing = []
                stable = []
                declining = []

                for concept in cluster_concepts:
                    if (self.concept_growth_predictions.get("all_predictions") and 
                        concept in self.concept_growth_predictions["all_predictions"]):
                        growth_rate = self.concept_growth_predictions["all_predictions"][concept]["growth_rate"]

                        if growth_rate > 0.2:
                            growing.append(concept)
                        elif growth_rate >= -0.1:
                            stable.append(concept)
                        else:
                            declining.append(concept)
                
                # Check for clusters where some concepts grow and others decline
                if growing and declining and len(growing) + len(declining) >= 3:
                    potential_gaps.append({
                        "cluster_id": cluster["cluster_id"],
                        "growing_concepts": growing,
                        "declining_concepts": declining,
                        "gap_description": ("This cluster shows uneven growth, suggesting "
                                          "potential research gaps where declining concepts "
                                          "could be reinvigorated through association with "
                                          "growing concepts.")
                    })
                    
            if potential_gaps:
                insights.append({
                    "insight_type": "research_gaps",
                    "description": "Identified potential research gaps based on uneven concept growth within clusters",
                    "gaps": potential_gaps
                })
                
        # Insight 3: Cross-disciplinary opportunities
        if self.identified_patterns["concept_clusters"] and len(self.identified_patterns["concept_clusters"]) >= 2:
            potential_connections = []
            # Look for concepts that belong to multiple clusters
            concept_to_cluster = defaultdict(list)

            for cluster in self.identified_patterns["concept_clusters"]:
                for concept in cluster["concepts"]:  # FIXED: Using "concepts" instead of "concept"
                    concept_to_cluster[concept].append(cluster["cluster_id"])
            
            # Find concepts that bridge multiple clusters
            bridge_concepts = {concept: clusters for concept, clusters in concept_to_cluster.items() if len(clusters) > 1}
            
            # Identify pairs of clusters that share concepts
            cluster_pairs = set()

            for concept, clusters in bridge_concepts.items():
                for i, c1 in enumerate(clusters):
                    for c2 in clusters[i+1:]:
                        if c1 != c2:
                            cluster_pairs.add((min(c1, c2), max(c1, c2), concept))
            
            # Convert to list of dictionary objects
            for c1, c2, concept in cluster_pairs:
                # Get cluster objects
                cluster1 = next((c for c in self.identified_patterns["concept_clusters"] if c["cluster_id"] == c1), None)
                cluster2 = next((c for c in self.identified_patterns["concept_clusters"] if c["cluster_id"] == c2), None)

                if cluster1 and cluster2:
                    potential_connections.append({
                        "clusters": [c1, c2],
                        "bridge_concept": concept,
                        "cluster1_concepts": cluster1["concepts"],
                        "cluster2_concepts": cluster2["concepts"],
                        "potential_description": (f"The concept '{concept}' bridges two distinct "
                                                f"research clusters, suggesting potential for cross-disciplinary "
                                                f"collaboration or innovative approaches.")
                    })
            
            if potential_connections:
                insights.append({
                    "insight_type": "cross_disciplinary_opportunities",
                    "description": "Identified potential cross-disciplinary research opportunities",
                    "connections": potential_connections
                })

        # Insight 4: Concept evolution patterns
        if self.concept_graph and hasattr(self.concept_graph, 'analyze_concept_evolution'):
            concept_evolution = self.concept_graph.analyze_concept_evolution()
            
            if concept_evolution and concept_evolution.get("emerging_concepts"):
                insights.append({
                    "insight_type": "concept_evolution",
                    "description": "Analyzed the evolution patterns of key concepts over time",
                    "emerging_concepts": concept_evolution["emerging_concepts"],
                    "consistent_concepts": concept_evolution.get("consistent_concepts", []),
                    "research_spans": {k: v for k, v in concept_evolution.get("research_span", {}).items() 
                                     if k in concept_evolution["emerging_concepts"] or 
                                        k in concept_evolution.get("consistent_concepts", [])}
                })
                
        return insights
