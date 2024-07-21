from pydantic import BaseModel, Field


class BaseTemplate(BaseModel):
    delim: str = Field(default="###\n\n", description="[template] delimiter")
    stop: str = Field(default="###", description="[template] stop sequences")

    train_context: str = Field(description="[template] context for feedback")
    train_instruction: str = Field(description="[template] instruction for feedback")
    train_regex: str = Field(description="[template] regular expression for parsing feedback responses")
    train_output: str = Field(description="[template] output format for feedback")

    test_context: str = Field(description="[template] context for generation")
    test_instruction: str = Field(description="[template] instruction for generation")
    test_regex: str = Field(description="[template] regular expression for parsing generation responses")
    test_output: str = Field(description="[template] output format for generation")


class AcronymTemplate(BaseTemplate):
    train_context: str = "Title: {title}\n\n"
    train_instruction: str = (
        "We want to create an acronym that effectively represents the given title. "
        "The acronym will be scored out of 10 points for each of the following four evaluation criteria:\n"
        "* Relevance to the title\n"
        "* Ease of pronunciation\n"
        "* Ease of spelling\n"
        "* Familiarity\n"
        "As there are four criteria, the acronym can receive a maximum score of 40 points.\n\n"
        "The first task is to create a bad acronym that does not align well with the four evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the four criteria and assign scores accordingly. "
        "The total evaluation score for this bad acronym should not exceed 20 points.\n"
        "The second task is to create a good acronym that aligns well with the four evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the four criteria and assign scores accordingly. "
        "The total evaluation score for this good acronym should be a minimum of 36 points.\n\n"
        "Answer in the following format:\n"
        "Bad acronym: <bad acronym>\n"
        "Feedback:\n"
        "* Relevance to the title: <feedback>. Score: <score>/10\n"
        "* Ease of pronunciation: <feedback>. Score: <score>/10\n"
        "* Ease of spelling: <feedback>. Score: <score>/10\n"
        "* Familiarity: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/40\n\n"
        "Good acronym: <good acronym>\n"
        "Feedback:\n"
        "* Relevance to the title: <feedback>. Score: <score>/10\n"
        "* Ease of pronunciation: <feedback>. Score: <score>/10\n"
        "* Ease of spelling: <feedback>. Score: <score>/10\n"
        "* Familiarity: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/40"
    )
    train_regex: str = r"Bad acronym:\s*(\w[\w\-]*)\s*.*?\n" r"Feedback:\s*\n" r"(?:\*|-) Relevance to the title:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Ease of pronunciation:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Ease of spelling:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Familiarity:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"Total score:\s*\d+/40\s*\n\n" r"Good acronym:\s*(\w[\w\-]*)\s*.*?\n" r"Feedback:\s*\n" r"(?:\*|-) Relevance to the title:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Ease of pronunciation:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Ease of spelling:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Familiarity:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    train_output: str = "Title: {title}\n\n" "Bad acronym: {acronym_a}\n" "Feedback:\n" "* Relevance to the title: {feedback_a.relevance} Score: {feedback_a.relevance_score}/10\n" "* Ease of pronunciation: {feedback_a.pronunciation} Score: {feedback_a.pronunciation_score}/10\n" "* Ease of spelling: {feedback_a.spelling} Score: {feedback_a.spelling_score}/10\n" "* Familiarity: {feedback_a.familiarity} Score: {feedback_a.familiarity_score}/10\n" "Total score: {feedback_a.total_score}/40\n\n" "Good acronym: {acronym_b}\n" "Feedback:\n" "* Relevance to the title: {feedback_b.relevance} Score: {feedback_b.relevance_score}/10\n" "* Ease of pronunciation: {feedback_b.pronunciation} Score: {feedback_b.pronunciation_score}/10\n" "* Ease of spelling: {feedback_b.spelling} Score: {feedback_b.spelling_score}/10\n" "* Familiarity: {feedback_b.familiarity} Score: {feedback_b.familiarity_score}/10\n" "Total score: {feedback_b.total_score}/40\n"

    test_context: str = "Title: {title}\n\n"
    test_instruction: str = "We want to create an acronym that effectively represents the given title. " "The acronym should meet the following four criteria:\n" "* Relevance to the title\n" "* Ease of pronunciation\n" "* Ease of spelling\n" "* Familiarity\n\n" "Answer in the following format:\n" "Acronym: <acronym>"
    test_regex: str = r"Acronym:\s*(\w[\w\-]*)\s*.*?(?:\n|$)"
    test_output: str = "Title: {title}\nAcronym: {acronym}\n"


class DialogTemplate(BaseTemplate):
    train_context: str = "Conversation history:\n{dialog}\n\n"
    train_instruction: str = (
        "We want to create a sensible and context-appropriate response for speaker B based on the given conversation history. "
        "The response will be scored out of 10 points for each of the following three evaluation criteria:\n"
        "* Consistency in conversational context and tone\n"
        "* Understanding the speaker's intent\n"
        "* Sustaining the conversation\n"
        "As there are three criteria, the response can receive a maximum score of 30 points.\n\n"
        "The first task is to create a bad response that does not align well with the three evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the three criteria and assign scores accordingly. "
        "The total evaluation score for this bad response should not exceed 15 points.\n"
        "The second task is to create a good response that aligns well with the three evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the three criteria and assign scores accordingly. "
        "The total evaluation score for this good response should be a minimum of 27 points.\n\n"
        "Answer in the following format:\n"
        "Bad response: <bad response>\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: <feedback>. Score: <score>/10\n"
        "* Understanding the speaker's intent: <feedback>. Score: <score>/10\n"
        "* Sustaining the conversation: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/30\n\n"
        "Good response: <good response>\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: <feedback>. Score: <score>/10\n"
        "* Understanding the speaker's intent: <feedback>. Score: <score>/10\n"
        "* Sustaining the conversation: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/30"
    )
    train_regex: str = r"Bad response:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Consistency in conversational context and tone:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Understanding the speaker's intent:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Sustaining the conversation:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"Total score:\s*\d+/30\s*\n\n" r"Good response:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Consistency in conversational context and tone:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Understanding the speaker's intent:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Sustaining the conversation:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    train_output: str = "Conversation history:\n{dialog}\n\n" "Bad response: {response_a}\n" "Feedback:\n" "* Consistency in conversational context and tone: {feedback_a.consistency} Score: {feedback_a.consistency_score}/10\n" "* Understanding the speaker's intent: {feedback_a.understand} Score: {feedback_a.understand_score}/10\n" "* Sustaining the conversation: {feedback_a.sustain} Score: {feedback_a.sustain_score}/10\n" "Total score: {feedback_a.total_score}/30\n\n" "Good response: {response_b}\n" "Feedback:\n" "* Consistency in conversational context and tone: {feedback_b.consistency} Score: {feedback_b.consistency_score}/10\n" "* Understanding the speaker's intent: {feedback_b.understand} Score: {feedback_b.understand_score}/10\n" "* Sustaining the conversation: {feedback_b.sustain} Score: {feedback_b.sustain_score}/10\n" "Total score: {feedback_b.total_score}/30\n"

    test_context: str = "Conversation history:\n{dialog}\n\n"
    test_instruction: str = "We want to create a sensible and context-appropriate response for speaker B based on the given conversation history. " "The response should meet the following three criteria:\n" "* Consistency in conversational context and tone\n" "* Understanding the speaker's intent\n" "* Sustaining the conversation\n\n" "Answer in the following format:\n" "Response: <response>"
    test_regex: str = r"Response:\s*(.*?)\s*(?:\n|$)"
    test_output: str = "Conversation history:\n{dialog}\nResponse: {response}\n"


class MathTemplate(BaseTemplate):
    train_context: str = "Math problem: {question}\n\n"
    train_instruction: str = (
        "We want to create a detailed solution for the given math problem. "
        "When solving the problem, break down the situation into the smallest logical steps where only one mathematical equation appears at a time. "
        'Approach the problem step by step, enclosing the required equations for each step with "<<" and ">>", and double-check whether the equation is correct. '
        'Present the final answer after "####".\n\n'
        "The following is an example of the desired solution for a math problem.\n"
        "Math problem: Chase and Rider can ride their bikes thrice a day for 5 days; but on two other days, they ride twice the times they do on usual days. How many times do they ride their bikes a week?\n"
        "Solution:\n"
        "Each person rides a bike a total of 3 * 5 = <<3*5=15>>15 times over 5 days.\n"
        "Therefore, they ride a total of 15 + 15 = <<15+15=30>>30 times over 5 days.\n"
        "Each person rides a bike 3 * 2 = <<3*2=6>>6 times each day for two days.\n"
        "This means that each person rides a total of 6 * 2 = <<6*2=12>>12 times over 2 days.\n"
        "Consequently, they ride a total of 12 + 12 = <<12+12=24>>24 times over 2 days.\n"
        "To summarize, they ride a total of 30 + 24 = <<30+24=54>>54 times over the course of one week.\n"
        "#### 54\n\n"
        "The solution will be scored out of 10 points for each of the following two evaluation criteria:\n"
        "* Adequacy of dividing the problem into logical steps\n"
        "* Validity of equations\n"
        "As there are two criteria, the solution can receive a maximum score of 20 points.\n\n"
        "The first task is to create a bad solution that does not align well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this bad solution should not exceed 10 points.\n"
        "The second task is to create a good solution that aligns well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this good solution should be a minimum of 18 points.\n\n"
        "Answer in the following format:\n"
        "Bad solution:\n<steps>\n#### <answer>\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: <feedback>. Score: <score>/10\n"
        "* Validity of equations: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20\n\n"
        "Good solution:\n<steps>\n#### <answer>\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: <feedback>. Score: <score>/10\n"
        "* Validity of equations: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    train_regex: str = r"Bad solution:\s*\n\s*((?:.|\n)+?)\s*\n\s*####\s*(\d+)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Adequacy of dividing the problem into logical steps:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Validity of equations:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"Total score:\s*\d+/20\s*\n\n" r"Good solution:\s*\n\s*((?:.|\n)+?)\s*\n\s*####\s*(\d+)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Adequacy of dividing the problem into logical steps:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Validity of equations:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    train_output: str = "Math problem: {question}\n\n" "Bad solution:\n{solution_a.steps}\n#### {solution_a.answer}\n" "Feedback:\n" "* Adequacy of dividing the problem into logical steps: {feedback_a.adequacy} Score: {feedback_a.adequacy_score}/10\n" "* Validity of equations: {feedback_a.validity} Score: {feedback_a.validity_score}/10\n" "Total score: {feedback_a.total_score}/20\n\n" "Good solution:\n{solution_b.steps}\n#### {solution_b.answer}\n" "Feedback:\n" "* Adequacy of dividing the problem into logical steps: {feedback_b.adequacy} Score: {feedback_b.adequacy_score}/10\n" "* Validity of equations: {feedback_b.validity} Score: {feedback_b.validity_score}/10\n" "Total score: {feedback_b.total_score}/20\n"

    test_context: str = "Math problem: {question}\n\n"
    test_instruction: str = (
        "We want to create a detailed solution for the given math problem. "
        "When solving the problem, break down the situation into the smallest logical steps where only one mathematical equation appears at a time. "
        'Approach the problem step by step, enclosing the required equations for each step with "<<" and ">>", and double-check whether the equation is correct. '
        'Present the final answer after "####".\n\n'
        "The following is an example of the desired solution for a math problem.\n"
        "Math problem: Chase and Rider can ride their bikes thrice a day for 5 days; but on two other days, they ride twice the times they do on usual days. How many times do they ride their bikes a week?\n"
        "Solution:\n"
        "Each person rides a bike a total of 3 * 5 = <<3*5=15>>15 times over 5 days.\n"
        "Therefore, they ride a total of 15 + 15 = <<15+15=30>>30 times over 5 days.\n"
        "Each person rides a bike 3 * 2 = <<3*2=6>>6 times each day for two days.\n"
        "This means that each person rides a total of 6 * 2 = <<6*2=12>>12 times over 2 days.\n"
        "Consequently, they ride a total of 12 + 12 = <<12+12=24>>24 times over 2 days.\n"
        "To summarize, they ride a total of 30 + 24 = <<30+24=54>>54 times over the course of one week.\n"
        "#### 54\n\n"
        "The solution should meet the following two criteria:\n"
        "* Adequacy of dividing the problem into logical steps\n"
        "* Validity of equations\n\n"
        "Answer in the following format:\n"
        "Solution:\n"
        "<steps>\n"
        "#### <answer>"
    )
    test_regex: str = r"Solution:\n\s*((?:.|\n)+?)\s*\n\s*####\s*(\d+)\s*(?:\n|$)"
    test_output: str = "Math problem: {question}\nSolution:\n{solution.steps}\n#### {solution.answer}\n"


class SentenceTemplate(BaseTemplate):
    train_context: str = "Concepts: {concepts}\n\n"
    train_instruction: str = (
        "We want to create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence. "
        "The sentence will be scored out of 10 points for each of the following two evaluation criteria:\n"
        "* Inclusion of the concepts\n"
        "* Logical coherence\n"
        "As there are two criteria, the sentence can receive a maximum score of 20 points.\n\n"
        "The first task is to create a bad sentence that does not align well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this bad sentence should not exceed 10 points.\n"
        "The second task is to create a good sentence that aligns well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this good sentence should be a minimum of 18 points.\n\n"
        "Answer in the following format:\n"
        "Bad sentence: <bad sentence>\n"
        "Feedback:\n"
        "* Inclusion of the concepts: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20\n\n"
        "Good sentence: <good sentence>\n"
        "Feedback:\n"
        "* Inclusion of the concepts: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    train_regex: str = r"Bad sentence:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Inclusion of the concepts:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"Total score:\s*\d+/20\s*\n\n" r"Good sentence:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Inclusion of the concepts:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    train_output: str = "Concepts: {concepts}\n\n" "Bad sentence: {sentence_a}\n" "Feedback:\n" "* Inclusion of the concepts: {feedback_a.inclusion} Score: {feedback_a.inclusion_score}/10\n" "* Logical coherence: {feedback_a.logical} Score: {feedback_a.logical_score}/10\n" "Total score: {feedback_a.total_score}/20\n\n" "Good sentence: {sentence_b}\n" "Feedback:\n" "* Inclusion of the concepts: {feedback_b.inclusion} Score: {feedback_b.inclusion_score}/10\n" "* Logical coherence: {feedback_b.logical} Score: {feedback_b.logical_score}/10\n" "Total score: {feedback_b.total_score}/20\n"

    test_context: str = "Concepts: {concepts}\n\n"
    test_instruction: str = "We want to create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence. " "The sentence should meet the following two criteria:\n" "* Inclusion of the concepts\n" "* Logical coherence\n\n" "Answer in the following format:\n" "Sentence: <sentence>"
    test_regex: str = r"Sentence:\s*(.*?)\s*(?:\n|$)"
    test_output: str = "Concepts: {concepts}\nSentence: {sentence}\n"


class SentimentTemplate(BaseTemplate):
    train_context: str = "Review: {review}\n\n"
    train_instruction: str = (
        "We want to create a reversed review that conveys the opposite sentiment of the given review. "
        "The reversed review will be scored out of 10 points for each of the following two evaluation criteria:\n"
        "* Effectiveness of sentiment reversal\n"
        "* Logical coherence\n"
        "As there are two criteria, the reversed review can receive a maximum score of 20 points.\n\n"
        "The first task is to create a bad example of reversed review that does not align well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this bad example should not exceed 10 points.\n"
        "The second task is to create a good example of reversed review that aligns well with the two evaluation criteria. "
        "Provide detailed feedback of two or more sentences for each of the two criteria and assign scores accordingly. "
        "The total evaluation score for this good example should be a minimum of 18 points.\n\n"
        "Answer in the following format:\n"
        "Bad example: <bad example>\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20\n\n"
        "Good example: <good example>\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    train_regex: str = r"Bad example:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Effectiveness of sentiment reversal:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"Total score:\s*\d+/20\s*\n\n" r"Good example:\s*(.*?)\s*\n" r"Feedback:\s*\n" r"(?:\*|-) Effectiveness of sentiment reversal:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    train_output: str = "Review: {review}\n\n" "Bad example: {reversed_review_a}\n" "Feedback:\n" "* Effectiveness of sentiment reversal: {feedback_a.effective} Score: {feedback_a.effective_score}/10\n" "* Logical coherence: {feedback_a.logical} Score: {feedback_a.logical_score}/10\n" "Total score: {feedback_a.total_score}/20\n\n" "Good example: {reversed_review_b}\n" "Feedback:\n" "* Effectiveness of sentiment reversal: {feedback_b.effective} Score: {feedback_b.effective_score}/10\n" "* Logical coherence: {feedback_b.logical} Score: {feedback_b.logical_score}/10\n" "Total score: {feedback_b.total_score}/20\n"

    test_context: str = "Review: {review}\n\n"
    test_instruction: str = "We want to create a reversed review that conveys the opposite sentiment of the given review. " "The reversed review should meet the following two criteria:\n" "* Effectiveness of sentiment reversal\n" "* Logical coherence\n\n" "Answer in the following format:\n" "Reversed review: <reversed review>"
    test_regex: str = r"Reversed review:\s*(.*?)\s*(?:\n|$)"
    test_output: str = "Review: {review}\nReversed review: {reversed_review}\n"


TEMPLATE = {
    "acronym": AcronymTemplate,
    "dialog": DialogTemplate,
    "math": MathTemplate,
    "sentence": SentenceTemplate,
    "sentiment": SentimentTemplate,
}


def get_template(args):
    if args.task not in TEMPLATE:
        raise ValueError(f"{args.task} is not supported.")
    return TEMPLATE[args.task](**vars(args))
