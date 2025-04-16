import os
import json
import logging
import google.generativeai as genai  

from error_handlers import BadRequest, InternalServerError

# Configure Gemini
genai.configure(api_key="AIzaSyB49wDyRv0wPscQ-urPgKYKBIS8jq8VZT8")  # Using environment variable properly
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def generate_crop_report(crop_name):
    try:
        prompt = f"""
        Given the crop {crop_name}, provide a detailed report in JSON format with the following sections:
        1. "idealRangeOfParams": Ideal ranges for soil parameters:
           - pH (e.g., 6.0–7.0)
           - Phosphorus (mg/kg or ppm, e.g., 20–40)
           - Nitrogen (mg/kg or ppm, e.g., 100–150)
           - Potash (mg/kg or ppm, e.g., 150–200)
        2. "growingTips": A list of 3–5 growing tips, each with:
           - "title": A short heading (e.g., "Water Management")
           - "description": 2–4 lines of practical advice in paragraph
        3. "growthTimeline": A map of growth stages with typical duration in days:
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