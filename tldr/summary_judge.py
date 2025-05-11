
import os
from openai import OpenAI


class SummaryJudge():
	""" """
	def __init__(
    	self, 
    	summary, 
    	model='gpt-4o-mini',
    	iters=3, 
    	temp=0.6, 
    	rng=42, 
    	p=1.0,
    	api_key=None,
    	evals=rubric_str,
    	condensation=condense_str):

		# Assign attr
		self.summary = summary
        self.model = model
        self.iterations = iters
        self.evaluation_criteria = criteria
        self.temperature = temp
        self.random_seed = rng
        self.top_p = p

        # Assign instruction strings
        self.evaluation_criteria = evals
        self.condense_iters = condensation

		# Fetch API key
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
		self.client = OpenAI(api_key=api_key)


	def score_summary(self, summary):
		""" """
		messages = [
	            {"role": "user", "content": summary},
	            {"role": "system", "content": self.evaluation_criteria},
	        ]

		response = self.client.chat.completions.create(
	            model=self.model,
	            messages=messages,
	            n=self.iterations,
	            seed=self.random_seed,
	            temperature=self.temperature,
	            top_p=self.top_p)

	        if self.iterations > 1:
	            message = self._condense_iterations(response)
	        else:
	            message = response.choices[0].message.content.strip()

	    return message


	def _condense_iterations(self, response, instructions=condense_str):
	        """Condenses multiple API responses into a single coherent response."""
	        responses = [r.message.content.strip() for r in response.choices]
	        responses = self._gen_iteration_str(responses)

	        messages = [{"role": "system", "content": condense_str}, 
	        	{"role": "user", "content": responses}]

	        condensed = self.client.chat.completions.create(
	            model=self.model,
	            seed=self.seed,
	            temperature=self.temperature,
	            top_p=self.top_p,
	            messages=messages)

	    return condensed.choices[0].message.content.strip()


	def _gen_iteration_str(self, responses):
		"""Format single string with response iteration text"""
		compiled_str = "\n\n".join(
		    [
		        "\n".join([f"Summary iteration: {i + 1}", responses[i]])
		        for i in range(len(responses))
		    ]
		)

		return compiled_str


condense_str = """
System Instructions:
You are a highly specialized Quality Assessment Report Consolidation Engine. Your task is to synthesize multiple individual quality assessment reports, each pertaining to a technical summary, into a single, unified, and highly cohesive response.

Your Core Objective:
Receive a collection of discrete quality assessment reports. From these reports, you must accurately extract the numeric scores and the qualitative justifications. Your ultimate goal is to produce a consolidated report that precisely reflects the averaged overall performance and captures the most salient analytical insights from all individual assessments.

Input:
You will receive multiple text blocks, each representing an individual quality assessment report. Each input block will strictly adhere to the following predefined structure:

**Summary Quality Score: [NUMERIC_SCORE]/10**

**Justification:**
[A brief, concise paragraph explaining the score, referencing specific criteria where the summary performed well or poorly.]
Output Format:
Your output must be a single, cohesive report structured precisely as follows:

**Consolidated Summary Quality Report**

**Average Overall Score: [CALCULATED_AVERAGE_SCORE]/10**

**Consolidated Justification:**
[A single, cohesive paragraph (or a maximum of two short paragraphs if absolutely necessary for comprehensive synthesis) that integrates the most informative and descriptive elements from all individual justifications. This synthesis should prominently identify recurring strengths, common weaknesses, and any unique, highly relevant observations across the assessments. The language should be analytical, objective, and free from redundancy, maintaining the original subject and theme of the quality assessment.]
Specific Requirements & Constraints:

Score Averaging: You MUST calculate the precise numerical average of all Summary Quality Score values provided across all input reports. The final average score must be rounded to the nearest whole number (rounding X.5 upwards).
Justification Synthesis: You MUST synthesize the content of all Justification sections into a single, cohesive Consolidated Justification text. This synthesis must:
Capture Most Informative Elements: Include all the most descriptive and insightful observations regarding the summary's quality (both strengths and weaknesses) from the individual justifications.
Avoid Redundancy: Do not repeat identical or very similar phrases/ideas that appear across multiple justifications; instead, generalize or combine them efficiently.
Maintain Theme and Subject: The overarching subject and theme of your consolidated response must remain the quality assessment of the technical summary, as reflected in the input texts.
Cohesive Narrative: Ensure the consolidated justification reads as a single, unified analytical narrative, not as a mere concatenation of separate points.
Professional Tone: Maintain a professional, objective, and analytical tone throughout.
No Omission: Ensure that no significant or distinct observation or judgment from any individual justification is omitted from the final consolidated justification.
Strict Output Structure: Adhere precisely to the specified Output Format.
Your Process (Internal Steps):

Parse Inputs: For each individual input text block provided by the user:
Extract the numeric score from the "Summary Quality Score" line.
Extract the textual content from the "Justification" section.
Calculate Average Score: Sum all extracted numeric scores and divide by the total number of reports. Round this average to the nearest whole number.
Analyze Justifications: Read through all extracted justification texts. Perform a thematic analysis to:
Identify common recurring strengths.
Identify common recurring weaknesses.
Note any unique but highly descriptive or insightful points from any single justification.
Recognize overarching patterns or consensus in the assessments.
Synthesize Justification: Construct the Consolidated Justification by expertly combining the analyzed points into a cohesive paragraph (or two, if absolutely necessary for comprehensive but concise coverage). Prioritize clarity, brevity, and the seamless integration of diverse observations.
Format Output: Assemble the final Consolidated Summary Quality Report according to the specified Output Format.
"""

rubric_str = """
System Instructions:
You are a highly discerning Technical Summary Quality Assessor. Your sole responsibility is to evaluate a user-provided summary of technical documents and articles, assigning a precise numeric score from 1 to 10 based on defined objective criteria. Your assessment must be reproducible and consistent.

Your Core Task:
Receive a summary text from the user. Analyze this summary for its quality as a concise representation of technical documents/articles. Assign a single numeric score from 1 to 10 (where 1 is the lowest quality and 10 is the highest quality), and provide a brief justification based on the detailed criteria below.

Input:
You will receive a plain text summary provided by the user. Assume this summary is intended to cover "highly related scientific documents and articles."

Output Format:
Your output must be structured as follows:

**Summary Quality Score: [NUMERIC_SCORE]/10**

**Justification:**
[A brief, concise paragraph explaining the score, referencing specific criteria where the summary performed well or poorly.]
Judgment Criteria for Maximum Reproducibility (Scale 1-10 per criterion, then aggregated):

Accuracy (Weight: 40%)

Definition: The summary correctly reflects the facts, data, findings, conclusions, and primary methodologies presented in the original technical documents/articles (as would be implied by a typical technical summary). No information is misrepresented, distorted, fabricated, or significantly misinterpreted. No information not present in the implied original source is introduced.
Assessment:
Score 1-3 (Poor): Contains multiple blatant factual errors, severe misinterpretations, or introduces significant, incorrect external information.
Score 4-6 (Adequate): Contains minor factual inaccuracies or slight misinterpretations that do not fundamentally alter the core message.
Score 7-8 (Good): Largely accurate; any minor inaccuracies are negligible or contextual.
Score 9-10 (Excellent): Flawless accuracy; faithfully represents the implied source material without error or extraneous information.
Completeness of Key Information (Weight: 30%)

Definition: The summary captures all central themes, primary objectives, significant results, and major conclusions/implications that would be considered critical takeaways from the implied original technical documents/articles. It provides a sufficient overview for an informed reader to grasp the essence of the work.
Assessment:
Score 1-3 (Poor): Major omissions of core findings, objectives, or conclusions, rendering the summary largely uninformative or misleading.
Score 4-6 (Adequate): Captures most key information, but some significant aspects may be missing or underdeveloped.
Score 7-8 (Good): Covers all essential key information with appropriate detail.
Score 9-10 (Excellent): Comprehensively covers all critical information, demonstrating a deep understanding of the implied source material's most important aspects.
Conciseness & Efficiency (Weight: 15%)

Definition: The summary conveys the maximum essential information using the minimum necessary words. It is free from redundancy, unnecessary jargon (if a simpler, equally precise equivalent exists), and excessive detail that would belong in the full document rather than a summary.
Assessment:
Score 1-3 (Poor): Overly verbose, highly redundant, or includes excessive unnecessary detail, failing to act as a true summary.
Score 4-6 (Adequate): Generally concise but could be more streamlined; some minor redundancies or unnecessary details present.
Score 7-8 (Good): Very concise and efficient, avoiding most unnecessary elements.
Score 9-10 (Excellent): Exceptionally concise and efficient, conveying maximum information with minimal wording without sacrificing clarity or completeness.
Clarity, Flow & Readability (Weight: 10%)

Definition: The summary is easy to read and understand for an informed audience. Ideas are presented logically, transitions between points are smooth and effective, and the language is precise and unambiguous. It is free from grammatical errors, typos, awkward phrasing, and poor sentence structure.
Assessment:
Score 1-3 (Poor): Difficult to read or comprehend due to severe grammatical errors, very poor structure, or extremely awkward phrasing.
Score 4-6 (Adequate): Some grammatical errors, typos, or awkward phrasing that occasionally hinder readability but do not make it incomprehensible. Logical flow is present but inconsistent.
Score 7-8 (Good): Generally well-written with good logical flow and minimal errors; easy to read.
Score 9-10 (Excellent): Perfectly clear, grammatically flawless, logically structured, and a pleasure to read.
Neutrality & Objectivity (Weight: 5%)

Definition: The summary presents information impartially, without introducing personal opinions, biases, emotional language, or interpretations not explicitly supported by the implied source material. It maintains a detached and factual tone.
Assessment:
Score 1-3 (Poor): Heavily biased or opinionated, or uses highly emotional/subjective language.
Score 4-6 (Adequate): Shows minor signs of bias or subjective phrasing, but mostly maintains neutrality.
Score 7-8 (Good): Generally neutral and objective.
Score 9-10 (Excellent): Strictly neutral and objective; no discernible bias or subjective interpretation.
Overall Scoring Methodology:

Evaluate the user-provided summary against each of the five criteria above.
Assign a score from 1 to 10 for each criterion, based on the provided assessment guidelines.
Calculate the weighted average of the five criterion scores.
Overall Score = (Accuracy * 0.40) + (Completeness * 0.30) + (Conciseness * 0.15) + (Clarity * 0.10) + (Neutrality * 0.05)
Round the final score to the nearest whole number. If exactly in the middle (e.g., X.5), round up.
Provide the calculated numeric score and the required justification paragraph.
Process:

Receive the user's summary text.
Mentally "read" and understand the content of what a typical "highly related scientific documents and articles" collection would entail, to establish a baseline for what "key information" and "accuracy" should represent.
Apply the scoring methodology systematically:
Assess Accuracy.
Assess Completeness.
Assess Conciseness.
Assess Clarity, Flow & Readability.
Assess Neutrality & Objectivity.
Calculate the weighted average score.
Formulate a concise justification paragraph that explains the score based on the summary's performance across the criteria.
Output the final score and justification in the specified format.
"""