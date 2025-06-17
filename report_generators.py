import os
import json
import logging
import google.generativeai as genai  

from error_handlers import BadRequest, InternalServerError

genai.configure(api_key="AIzaSyA20c4NyuDxj1CczvWj-c5vO1B97QYb5bo")  
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def generate_crop_report(crop_name):
    try:
        prompt = f"""
        Given the crop {crop_name}, provide a detailed report in JSON format with the following sections:
        1. "idealRangeOfParams": Ideal ranges for soil parameters:
           - Temperature (°C, e.g., 20–30) 
           - pH (e.g., 6.0–7.0)
           - Phosphorus (mg/kg , e.g., 20–40)
           - Nitrogen (mg/kg , e.g., 100–150)
           - Potash (mg/kg , e.g., 150–200)
        2. "growingTips": A list of 3–5 growing tips, each with:
           - "title": A short heading (e.g., "Water Management")
           - "description": 2–4 lines of practical advice in paragraph
        3. "growthTimeline": A map of growth stages with typical duration in days (it should be in order of growth stages):
           - "Seedling": e.g., 7–14 days
           - "Vegetative": e.g., 30–60 days
           - "Reproductive": e.g., 20–30 days
           - "Ripening": e.g., 15–25 days
        Ensure ranges are crop-specific, tips are actionable, and timelines are realistic. Return only the JSON object.
        """
        logging.info(f"Generating report for crop: {crop_name}")
        print
        response = gemini_model.generate_content(prompt)
        print('This is the response of the crop recommendation report:', response)
        
        # Check if response has content
        if not hasattr(response, 'text'):
            # For newer Gemini API versions
            report = response.parts[0].text
        else:
            # For older Gemini API versions
            report = response.text
            
        report = report.strip()
        
        # Parse JSON response (Gemini may wrap in ```json)
        if "```json" in report:
            # Extract JSON content between markdown code blocks
            start_idx = report.find("```json") + 7
            end_idx = report.find("```", start_idx)
            if end_idx != -1:
                report = report[start_idx:end_idx].strip()
            else:
                report = report[start_idx:].strip()
        elif report.startswith("```") and report.endswith("```"):
            # Handle case where language isn't specified
            report = report[3:-3].strip()
            
        # Parse the JSON
        try:
            return json.loads(report)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parsing error: {str(json_err)}")
            logging.error(f"Raw response: {report}")
            raise BadRequest(f"Failed to parse Gemini response as JSON: {str(json_err)}")
            
    except Exception as e:
        logging.error(f"Gemini report error: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to generate crop report: {str(e)}")

def generate_fertilizer_report(fertilizer_name, crop_type):
    try:
        prompt = f"""
        Given the fertilizer '{fertilizer_name}' and crop '{crop_type}', provide a detailed report in JSON format with the following sections:
        1. "fertilizerDescription": A concise one- or two-word description of the fertilizer (e.g., "Potassium-rich" for MOP, "Nitrogen-based" for Urea).
        2. "applicationRate": Recommended application rate (e.g., "100 kg/hectare").
        3. "method": Application method (e.g., "Broadcast and mix well with soil").
        4. "timing": Best time to apply (e.g., "Apply before planting or during growth stage").
        5. "importantNote": A key precaution or tip (e.g., "Avoid over-application to prevent soil imbalance").
        Ensure the report is specific to the fertilizer and crop, with practical and realistic recommendations. Return only the JSON object.
        """
        logging.info(f"Generating fertilizer report for {fertilizer_name} and crop {crop_type}")
        print(f"Generating fertilizer report for {fertilizer_name} and crop {crop_type}")
        response = gemini_model.generate_content(prompt)
        print('This is the response of the fertilizer recommendation report:', response)
        
        # Check if response has content
        if not hasattr(response, 'text'):
            report = response.parts[0].text
        else:
            report = response.text
            
        report = report.strip()
        
        # Parse JSON response (handle markdown code blocks)
        if "```json" in report:
            start_idx = report.find("```json") + 7
            end_idx = report.find("```", start_idx)
            if end_idx != -1:
                report = report[start_idx:end_idx].strip()
            else:
                report = report[start_idx:].strip()
        elif report.startswith("```") and report.endswith("```"):
            report = report[3:-3].strip()
            
        # Parse the JSON
        try:
            return json.loads(report)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parsing error: {str(json_err)}")
            logging.error(f"Raw response: {report}")
            raise BadRequest(f"Failed to parse Gemini response as JSON: {str(json_err)}")
            
    except Exception as e:
        logging.error(f"Gemini fertilizer report error: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to generate fertilizer report: {str(e)}")
def generate_growth_report(features, prediction):
    try:
        prompt = f"""
        Given the crop growth prediction '{prediction}' (either 'Growable' or 'Not Growable') and the following input parameters:
        - Crop: {features['Crop']}
        - Soil Type: {features['Soil_Type']}
        - Sunlight Hours: {features['Sunlight_Hours']} hours
        - Temperature: {features['Temperature']}°C
        - Humidity: {features['Humidity']}%
        - Water Frequency: {features['Water_Frequency']}
        - Fertilizer Type: {features['Fertilizer_Type']}
        Provide a list of 3–5 practical recommendations as strings to improve or maintain crop growth. If 'Growable', suggest ways to optimize conditions. If 'Not Growable', suggest corrections to make the crop growable. Return only a JSON array of strings, e.g., ["Adjust irrigation to weekly intervals", "Increase sunlight exposure"].
        """
        logging.info(f"Generating growth report for {features['Crop']} with prediction {prediction}")
        response = gemini_model.generate_content(prompt)
        
        # Check if response has content
        if not hasattr(response, 'text'):
            report = response.parts[0].text
        else:
            report = response.text
            
        report = report.strip()
        
        # Parse JSON response
        if "```json" in report:
            start_idx = report.find("```json") + 7
            end_idx = report.find("```", start_idx)
            if end_idx != -1:
                report = report[start_idx:end_idx].strip()
            else:
                report = report[start_idx:].strip()
        elif report.startswith("```") and report.endswith("```"):
            report = report[3:-3].strip()
            
        try:
            recommendations = json.loads(report)
            if not isinstance(recommendations, list) or not all(isinstance(r, str) for r in recommendations):
                raise ValueError("Expected a list of strings")
            return recommendations
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parsing error: {str(json_err)}")
            logging.error(f"Raw response: {report}")
            raise BadRequest(f"Failed to parse Gemini response as JSON: {str(json_err)}")
        except ValueError as ve:
            logging.error(f"Invalid response format: {str(ve)}")
            logging.error(f"Raw response: {report}")
            raise BadRequest(f"Invalid Gemini response format: {str(ve)}")
            
    except Exception as e:
        logging.error(f"Gemini growth report error: {str(e)}", exc_info=True)
        # Fallback recommendations
        fallback = {
            "Growable": [
                f"Maintain regular irrigation for {features['Crop']}",
                "Monitor soil nutrients periodically",
                f"Ensure {features['Sunlight_Hours']} hours of sunlight daily"
            ],
            "Not Growable": [
                f"Adjust {features['Crop']} irrigation to match soil needs",
                f"Optimize sunlight for {features['Crop']} to 6–8 hours",
                "Consult a local agronomist for soil correction"
            ]
        }
        logging.info(f"Using fallback recommendations for {prediction}")
        return fallback[prediction]
