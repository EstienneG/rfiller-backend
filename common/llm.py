from common.api_global_variables import api_global_variables


def summarize(rfp_title: str, rfp_content: str) -> str:
    rfp_summary = (
        api_global_variables.llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Tu es un agent qui résume des appels d'offre. Tu vas résumer et montrer les dates et les critères principaux d'évaluation de l'appel d'offre."
                    + "Titre de l'appel d'offre: "
                    + rfp_title
                    + "Contenu de l'appel d'offre: "
                    + rfp_content,
                }
            ],
            model="llama3-8b-8192",
        )
        .choices[0]
        .message.content
    )

    print(rfp_summary)
    return rfp_summary


def call_groq(question: str) -> str:
    return (
        api_global_variables.llm.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama3-8b-8192",
        )
        .choices[0]
        .message.content
    )
