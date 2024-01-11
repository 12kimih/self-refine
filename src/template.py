from pydantic import Field, BaseModel


class BaseTemplate(BaseModel):
    delim: str = Field(default="###\n\n", description="[template] delimiter")
    stop: str = Field(default="###", description="[template] stop sequences")
    sign: str = Field(default="<stop>", description="[template] stop sign")

    initial_context: str = Field(description="[template] context for initial generation")
    initial_instruction: str = Field(description="[template] instruction for initial generation")
    initial_query: str = Field(description="[template] query for initial generation")
    initial_response: str = Field(description="[template] response for initial generation")
    initial_regex: str = Field(description="[template] regular expression for parsing initial generation responses")
    initial_output: str = Field(description="[template] output format for initial generation")

    feedback_context: str = Field(description="[template] context for feedback")
    feedback_instruction: str = Field(description="[template] instruction for feedback")
    feedback_query: str = Field(description="[template] query for feedback")
    feedback_response: str = Field(description="[template] response for feedback")
    feedback_regex: str = Field(description="[template] regular expression for parsing feedback responses")
    feedback_output: str = Field(description="[template] output format for feedback")

    refine_context_0: str = Field(description="[template] context for refine 0")
    refine_context_1: str = Field(description="[template] context for refine 1")
    refine_context_2: str = Field(description="[template] context for refine 2")
    refine_instruction_0: str = Field(description="[template] instruction for refine 0")
    refine_instruction_1: str = Field(description="[template] instruction for refine 1")
    refine_instruction_2: str = Field(description="[template] instruction for refine 2")
    refine_instruction_auto: str = Field(description="[template] instruction for autonomous determination of stopping criteria")
    refine_query: str = Field(description="[template] query for refine")
    refine_response_0: str = Field(description="[template] response for refine 0")
    refine_response_1: str = Field(description="[template] response for refine 1")
    refine_response_2: str = Field(description="[template] response for refine 2")

    evaluation_context: str = Field(description="[template] context for evaluation")
    evaluation_instruction: str = Field(description="[template] instruction for evaluation")
    evaluation_output: str = Field(description="[template] output format for evaluation")


class AcronymTemplate(BaseTemplate):
    initial_context: str = "Title: {title}\n\n"
    initial_instruction: str = "Create an acronym that effectively represents the given title.\n\n"
    initial_query: str = "Acronym:"
    initial_response: str = "Acronym: {acronym}\n\n"
    initial_regex: str = r"^\s*(\w[\w\-]*|<stop>)\s*.*?(?:\n|$)"
    initial_output: str = "{n} Title: {title}\n{n} Acronym: {acronym}"

    feedback_context: str = "Title: {title}\n\nAcronym: {acronym}\n\n"
    feedback_instruction: str = (
        "Provide detailed feedback of two or more sentences for each of the following four evaluation criteria: relevance to the title, ease of pronunciation, ease of spelling, familiarity. "
        "Assign a score out of 10 points for each criterion.\n\n"
    )
    feedback_query: str = "Feedback:\n"
    feedback_response: str = (
        "Feedback:\n"
        "* Relevance to the title: {relevance} Score: {relevance_score}/10\n"
        "* Ease of pronunciation: {pronunciation} Score: {pronunciation_score}/10\n"
        "* Ease of spelling: {spelling} Score: {spelling_score}/10\n"
        "* Familiarity: {familiarity} Score: {familiarity_score}/10\n"
        "Total score: {total_score}/40\n\n"
    )
    feedback_regex: str = (
        r"(?:\*|-) Relevance to the title:\s*(.*?)\s*Score:\s*(\d+)/10\s*\n"
        r"(?:\*|-) Ease of pronunciation:\s*(.*?)\s*Score:\s*(\d+)/10\s*\n"
        r"(?:\*|-) Ease of spelling:\s*(.*?)\s*Score:\s*(\d+)/10\s*\n"
        r"(?:\*|-) Familiarity:\s*(.*?)\s*Score:\s*(\d+)/10\s*(?:\n|$)"
    )
    feedback_output: str = (
        "{n} Feedback:\n"
        "* Relevance to the title: {relevance} Score: {relevance_score}/10\n"
        "* Ease of pronunciation: {pronunciation} Score: {pronunciation_score}/10\n"
        "* Ease of spelling: {spelling} Score: {spelling_score}/10\n"
        "* Familiarity: {familiarity} Score: {familiarity_score}/10\n"
        "{n} Total score: {total_score}/40\n"
    )

    refine_context_0: str = (
        "Title: {title}\n\n"
        "Previous acronym: {acronym}\n"
        "Feedback:\n"
        "* Relevance to the title: {relevance} Score: {relevance_score}/10\n"
        "* Ease of pronunciation: {pronunciation} Score: {pronunciation_score}/10\n"
        "* Ease of spelling: {spelling} Score: {spelling_score}/10\n"
        "* Familiarity: {familiarity} Score: {familiarity_score}/10\n"
        "Total score: {total_score}/40\n\n"
    )
    refine_context_1: str = "Title: {title}\n\nPrevious acronym: {acronym}\n\n"
    refine_context_2: str = "Title: {title}\n\nPrevious acronym: {acronym}\n\n"
    refine_instruction_0: str = (
        "Create an acronym that effectively represents the given title. "
        "Make sure to create a new acronym distinct from the previous one. "
        "The new acronym should better align with the four evaluation criteria by incorporating feedback from the previous acronym."
    )
    refine_instruction_1: str = (
        "Create an acronym that effectively represents the given title. "
        "Make sure to create a new acronym distinct from the previous one. "
        "The new acronym should better align with the following four evaluation criteria: relevance to the title, ease of pronunciation, ease of spelling, familiarity."
    )
    refine_instruction_2: str = "Create an acronym that effectively represents the given title. " "Make sure to create a new acronym distinct from the previous one."
    refine_instruction_auto: str = ' If there is no room for improvement from the previous acronym, output "<stop>".'
    refine_query: str = "New acronym:"
    refine_response_0: str = (
        "New acronym: {acronym}\n"
        "Feedback:\n"
        "* Relevance to the title: {relevance} Score: {relevance_score}/10\n"
        "* Ease of pronunciation: {pronunciation} Score: {pronunciation_score}/10\n"
        "* Ease of spelling: {spelling} Score: {spelling_score}/10\n"
        "* Familiarity: {familiarity} Score: {familiarity_score}/10\n"
        "Total score: {total_score}/40\n\n"
    )
    refine_response_1: str = "New acronym: {acronym}\n\n"
    refine_response_2: str = "New acronym: {acronym}\n\n"

    evaluation_context: str = "Title: {title}\n\nAcronym: {acronym}\n\n"
    evaluation_instruction: str = (
        "We have created an acronym that effectively represents the given title.\n"
        "Provide detailed feedback of two or more sentences for each of the following four evaluation criteria: relevance to the title, ease of pronunciation, ease of spelling, familiarity.\n"
        "Assign a score out of 10 points for each criterion.\n\n"
        "Answer in the following format:\n"
        "Feedback:\n"
        "* Relevance to the title: <feedback>. Score: <score>/10\n"
        "* Ease of pronunciation: <feedback>. Score: <score>/10\n"
        "* Ease of spelling: <feedback>. Score: <score>/10\n"
        "* Familiarity: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/40"
    )
    evaluation_output: str = (
        "Title: {title}\n"
        "Acronym: {acronym}\n"
        "Feedback:\n"
        "* Relevance to the title: {relevance} Score: {relevance_score}/10\n"
        "* Ease of pronunciation: {pronunciation} Score: {pronunciation_score}/10\n"
        "* Ease of spelling: {spelling} Score: {spelling_score}/10\n"
        "* Familiarity: {familiarity} Score: {familiarity_score}/10\n"
        "Total score: {total_score}/40\n"
    )


class DialogTemplate(BaseTemplate):
    initial_context: str = "Conversation history:\n{dialog}\n\n"
    initial_instruction: str = "Create a sensible and context-appropriate response for speaker B based on the given conversation history.\n\n"
    initial_query: str = "Response:"
    initial_response: str = "Response: {response}\n\n"
    initial_regex: str = r"^\s*(.*?)\s*(?:\n|$)"
    initial_output: str = "{n} Conversation history:\n{dialog}\n{n} Response: {response}"

    feedback_context: str = "Conversation history:\n{dialog}\n\nResponse: {response}\n\n"
    feedback_instruction: str = (
        "Provide detailed feedback of two or more sentences for each of the following three evaluation criteria: consistency in conversational context and tone, understanding the speaker's intent, sustaining the conversation. "
        "Assign a score out of 10 points for each criterion.\n\n"
    )
    feedback_query: str = "Feedback:\n"
    feedback_response: str = (
        "Feedback:\n"
        "* Consistency in conversational context and tone: {consistency} Score: {consistency_score}/10\n"
        "* Understanding the speaker's intent: {understand} Score: {understand_score}/10\n"
        "* Sustaining the conversation: {sustain} Score: {sustain_score}/10\n"
        "Total score: {total_score}/30\n\n"
    )
    feedback_regex: str = (
        r"(?:\*|-) Consistency in conversational context and tone:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n"
        r"(?:\*|-) Understanding the speaker's intent:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n"
        r"(?:\*|-) Sustaining the conversation:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    )
    feedback_output: str = (
        "{n} Feedback:\n"
        "* Consistency in conversational context and tone: {consistency} Score: {consistency_score}/10\n"
        "* Understanding the speaker's intent: {understand} Score: {understand_score}/10\n"
        "* Sustaining the conversation: {sustain} Score: {sustain_score}/10\n"
        "{n} Total score: {total_score}/30\n"
    )

    refine_context_0: str = (
        "Conversation history:\n{dialog}\n\n"
        "Previous response: {response}\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: {consistency} Score: {consistency_score}/10\n"
        "* Understanding the speaker's intent: {understand} Score: {understand_score}/10\n"
        "* Sustaining the conversation: {sustain} Score: {sustain_score}/10\n"
        "Total score: {total_score}/30\n\n"
    )
    refine_context_1: str = "Conversation history:\n{dialog}\n\nPrevious response: {response}\n\n"
    refine_context_2: str = "Conversation history:\n{dialog}\n\nPrevious response: {response}\n\n"
    refine_instruction_0: str = (
        "Create a sensible and context-appropriate response for speaker B based on the given conversation history. "
        "Make sure to create a new response distinct from the previous one. "
        "The new response should better align with the three evaluation criteria by incorporating feedback from the previous response."
    )
    refine_instruction_1: str = (
        "Create a sensible and context-appropriate response for speaker B based on the given conversation history. "
        "Make sure to create a new response distinct from the previous one. "
        "The new response should better align with the following three evaluation criteria: consistency in conversational context and tone, understanding the speaker's intent, sustaining the conversation."
    )
    refine_instruction_2: str = (
        "Create a sensible and context-appropriate response for speaker B based on the given conversation history. " "Make sure to create a new response distinct from the previous one."
    )
    refine_instruction_auto: str = ' If there is no room for improvement from the previous response, output "<stop>".'
    refine_query: str = "New response:"
    refine_response_0: str = (
        "New response: {response}\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: {consistency} Score: {consistency_score}/10\n"
        "* Understanding the speaker's intent: {understand} Score: {understand_score}/10\n"
        "* Sustaining the conversation: {sustain} Score: {sustain_score}/10\n"
        "Total score: {total_score}/30\n\n"
    )
    refine_response_1: str = "New response: {response}\n\n"
    refine_response_2: str = "New response: {response}\n\n"

    evaluation_context: str = "Conversation history:\n{dialog}\n\nResponse: {response}\n\n"
    evaluation_instruction: str = (
        "We have created a sensible and context-appropriate response for speaker B based on the given conversation history.\n"
        "Provide detailed feedback of two or more sentences for each of the following three evaluation criteria: consistency in conversational context and tone, understanding the speaker's intent, sustaining the conversation.\n"
        "Assign a score out of 10 points for each criterion.\n\n"
        "Answer in the following format:\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: <feedback>. Score: <score>/10\n"
        "* Understanding the speaker's intent: <feedback>. Score: <score>/10\n"
        "* Sustaining the conversation: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/30"
    )
    evaluation_output: str = (
        "Conversation history:\n{dialog}\n"
        "Response: {response}\n"
        "Feedback:\n"
        "* Consistency in conversational context and tone: {consistency} Score: {consistency_score}/10\n"
        "* Understanding the speaker's intent: {understand} Score: {understand_score}/10\n"
        "* Sustaining the conversation: {sustain} Score: {sustain_score}/10\n"
        "Total score: {total_score}/30\n"
    )


class MathTemplate(BaseTemplate):
    initial_context: str = "Math problem: {question}\n\n"
    initial_instruction: str = "Create a detailed solution for the given math problem.\n\n"
    initial_query: str = "Solution:"
    initial_response: str = "Solution:\n{solution.steps}\n#### {solution.answer}\n\n"
    initial_regex: str = r"^\s*((?:.|\n)+?)\s*\n\s*####\s*(\d+)\s*(?:\n|$)"
    initial_output: str = "{n} Math problem: {question}\n{n} Solution:\n{solution.steps}\n#### {solution.answer}"

    feedback_context: str = "Math problem: {question}\n\nSolution:\n{solution.steps}\n#### {solution.answer}\n\n"
    feedback_instruction: str = (
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: adequacy of dividing the problem into logical steps, validity of equations. "
        "Assign a score out of 10 points for each criterion.\n\n"
    )
    feedback_query: str = "Feedback:\n"
    feedback_response: str = (
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: {adequacy} Score: {adequacy_score}/10\n"
        "* Validity of equations: {validity} Score: {validity_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    feedback_regex: str = r"(?:\*|-) Adequacy of dividing the problem into logical steps:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Validity of equations:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    feedback_output: str = (
        "{n} Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: {adequacy} Score: {adequacy_score}/10\n"
        "* Validity of equations: {validity} Score: {validity_score}/10\n"
        "{n} Total score: {total_score}/20\n"
    )

    refine_context_0: str = (
        "Math problem: {question}\n\n"
        "Previous solution:\n{solution.steps}\n#### {solution.answer}\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: {adequacy} Score: {adequacy_score}/10\n"
        "* Validity of equations: {validity} Score: {validity_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_context_1: str = "Math problem: {question}\n\nPrevious solution:\n{solution.steps}\n#### {solution.answer}\n\n"
    refine_context_2: str = "Math problem: {question}\n\nPrevious solution:\n{solution.steps}\n#### {solution.answer}\n\n"
    refine_instruction_0: str = (
        "Create a detailed solution for the given math problem. "
        "Make sure to create a new solution distinct from the previous one. "
        "The new solution should better align with the two evaluation criteria by incorporating feedback from the previous solution."
    )
    refine_instruction_1: str = (
        "Create a detailed solution for the given math problem. "
        "Make sure to create a new solution distinct from the previous one. "
        "The new solution should better align with the following two evaluation criteria: adequacy of dividing the problem into logical steps, validity of equations."
    )
    refine_instruction_2: str = "Create a detailed solution for the given math problem. " "Make sure to create a new solution distinct from the previous one."
    refine_instruction_auto: str = ' If there is no room for improvement from the previous solution, output "<stop>".'
    refine_query: str = "New solution:"
    refine_response_0: str = (
        "New solution:\n{solution.steps}\n#### {solution.answer}\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: {adequacy} Score: {adequacy_score}/10\n"
        "* Validity of equations: {validity} Score: {validity_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_response_1: str = "New solution:\n{solution.steps}\n#### {solution.answer}\n\n"
    refine_response_2: str = "New solution:\n{solution.steps}\n#### {solution.answer}\n\n"

    evaluation_context: str = "Math problem: {question}\n\nSolution:\n{solution.steps}\n#### {solution.answer}\n\n"
    evaluation_instruction: str = (
        "We have created a detailed solution for the given math problem.\n"
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: adequacy of dividing the problem into logical steps, validity of equations.\n"
        "Assign a score out of 10 points for each criterion.\n\n"
        "Answer in the following format:\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: <feedback>. Score: <score>/10\n"
        "* Validity of equations: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    evaluation_output: str = (
        "Math problem: {question}\n"
        "Solution:\n{solution.steps}\n#### {solution.answer}\n"
        "Feedback:\n"
        "* Adequacy of dividing the problem into logical steps: {adequacy} Score: {adequacy_score}/10\n"
        "* Validity of equations: {validity} Score: {validity_score}/10\n"
        "Total score: {total_score}/20\n"
    )


class SentenceTemplate(BaseTemplate):
    initial_context: str = "Concepts: {concepts}\n\n"
    initial_instruction: str = "Create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence.\n\n"
    initial_query: str = "Sentence:"
    initial_response: str = "Sentence: {sentence}\n\n"
    initial_regex: str = r"^\s*(.*?)\s*(?:\n|$)"
    initial_output: str = "{n} Concepts: {concepts}\n{n} Sentence: {sentence}"

    feedback_context: str = "Concepts: {concepts}\n\nSentence: {sentence}\n\n"
    feedback_instruction: str = (
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: inclusion of the concepts, logical coherence. "
        "Assign a score out of 10 points for each criterion.\n\n"
    )
    feedback_query: str = "Feedback:\n"
    feedback_response: str = (
        "Feedback:\n" "* Inclusion of the concepts: {inclusion} Score: {inclusion_score}/10\n" "* Logical coherence: {logical} Score: {logical_score}/10\n" "Total score: {total_score}/20\n\n"
    )
    feedback_regex: str = r"(?:\*|-) Inclusion of the concepts:\s*(.*?)\s*Score:\s*(\d+)/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)/10\s*(?:\n|$)"
    feedback_output: str = (
        "{n} Feedback:\n" "* Inclusion of the concepts: {inclusion} Score: {inclusion_score}/10\n" "* Logical coherence: {logical} Score: {logical_score}/10\n" "{n} Total score: {total_score}/20\n"
    )

    refine_context_0: str = (
        "Concepts: {concepts}\n\n"
        "Previous sentence: {sentence}\n"
        "Feedback:\n"
        "* Inclusion of the concepts: {inclusion} Score: {inclusion_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_context_1: str = "Concepts: {concepts}\n\nPrevious sentence: {sentence}\n\n"
    refine_context_2: str = "Concepts: {concepts}\n\nPrevious sentence: {sentence}\n\n"
    refine_instruction_0: str = (
        "Create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence. "
        "Make sure to create a new sentence distinct from the previous one. "
        "The new sentence should better align with the two evaluation criteria by incorporating feedback from the previous sentence."
    )
    refine_instruction_1: str = (
        "Create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence. "
        "Make sure to create a new sentence distinct from the previous one. "
        "The new sentence should better align with the following two evaluation criteria: inclusion of the concepts, logical coherence."
    )
    refine_instruction_2: str = (
        "Create a sentence that incorporates as many given concepts as possible within the bounds of logical coherence. " "Make sure to create a new sentence distinct from the previous one."
    )
    refine_instruction_auto: str = ' If there is no room for improvement from the previous sentence, output "<stop>".'
    refine_query: str = "New sentence:"
    refine_response_0: str = (
        "New sentence: {sentence}\n"
        "Feedback:\n"
        "* Inclusion of the concepts: {inclusion} Score: {inclusion_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_response_1: str = "New sentence: {sentence}\n\n"
    refine_response_2: str = "New sentence: {sentence}\n\n"

    evaluation_context: str = "Concepts: {concepts}\n\nSentence: {sentence}\n\n"
    evaluation_instruction: str = (
        "We have created a sentence that incorporates as many given concepts as possible within the bounds of logical coherence.\n"
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: inclusion of the concepts, logical coherence.\n"
        "Assign a score out of 10 points for each criterion.\n\n"
        "Answer in the following format:\n"
        "Feedback:\n"
        "* Inclusion of the concepts: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    evaluation_output: str = (
        "Concepts: {concepts}\n"
        "Sentence: {sentence}\n"
        "Feedback:\n"
        "* Inclusion of the concepts: {inclusion} Score: {inclusion_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n"
    )


class SentimentTemplate(BaseTemplate):
    initial_context: str = "Review: {review}\n\n"
    initial_instruction: str = "Create a reversed review that conveys the opposite sentiment of the given review.\n\n"
    initial_query: str = "Reversed review:"
    initial_response: str = "Reversed review: {reversed_review}\n\n"
    initial_regex: str = r"^\s*(.*?)\s*(?:\n|$)"
    initial_output: str = "{n} Review: {review}\n{n} Reversed review: {reversed_review}"

    feedback_context: str = "Review: {review}\n\nReversed review: {reversed_review}\n\n"
    feedback_instruction: str = (
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: effectiveness of sentiment reversal, logical coherence. "
        "Assign a score out of 10 points for each criterion.\n\n"
    )
    feedback_query: str = "Feedback:\n"
    feedback_response: str = (
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: {effective} Score: {effective_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    feedback_regex: str = r"(?:\*|-) Effectiveness of sentiment reversal:\s*(.*?)\s*Score:\s*(\d+)\/10\s*\n" r"(?:\*|-) Logical coherence:\s*(.*?)\s*Score:\s*(\d+)\/10\s*(?:\n|$)"
    feedback_output: str = (
        "{n} Feedback:\n"
        "* Effectiveness of sentiment reversal: {effective} Score: {effective_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "{n} Total score: {total_score}/20\n"
    )

    refine_context_0: str = (
        "Review: {review}\n\n"
        "Previous reversed review: {reversed_review}\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: {effective} Score: {effective_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_context_1: str = "Review: {review}\n\nPrevious reversed review: {reversed_review}\n\n"
    refine_context_2: str = "Review: {review}\n\nPrevious reversed review: {reversed_review}\n\n"
    refine_instruction_0: str = (
        "Create a reversed review that conveys the opposite sentiment of the given review. "
        "Make sure to create a new reversed review distinct from the previous one. "
        "The new reversed review should better align with the two evaluation criteria by incorporating feedback from the previous reversed review."
    )
    refine_instruction_1: str = (
        "Create a reversed review that conveys the opposite sentiment of the given review. "
        "Make sure to create a new reversed review distinct from the previous one. "
        "The new reversed review should better align with the following two evaluation criteria: effectiveness of sentiment reversal, logical coherence."
    )
    refine_instruction_2: str = "Create a reversed review that conveys the opposite sentiment of the given review. " "Make sure to create a new reversed review distinct from the previous one."
    refine_instruction_auto: str = ' If there is no room for improvement from the previous reversed review, output "<stop>".'
    refine_query: str = "New reversed review:"
    refine_response_0: str = (
        "New reversed review: {reversed_review}\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: {effective} Score: {effective_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n\n"
    )
    refine_response_1: str = "New reversed review: {reversed_review}\n\n"
    refine_response_2: str = "New reversed review: {reversed_review}\n\n"

    evaluation_context: str = "Review: {review}\n\nReversed review: {reversed_review}\n\n"
    evaluation_instruction: str = (
        "We have created a reversed review that conveys the opposite sentiment of the given review.\n"
        "Provide detailed feedback of two or more sentences for each of the following two evaluation criteria: effectiveness of sentiment reversal, logical coherence.\n"
        "Assign a score out of 10 points for each criterion.\n\n"
        "Answer in the following format:\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: <feedback>. Score: <score>/10\n"
        "* Logical coherence: <feedback>. Score: <score>/10\n"
        "Total score: <total score>/20"
    )
    evaluation_output: str = (
        "Review: {review}\n"
        "Reversed review: {reversed_review}\n"
        "Feedback:\n"
        "* Effectiveness of sentiment reversal: {effective} Score: {effective_score}/10\n"
        "* Logical coherence: {logical} Score: {logical_score}/10\n"
        "Total score: {total_score}/20\n"
    )


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
