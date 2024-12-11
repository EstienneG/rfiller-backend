import phospho
from common.api_global_variables import api_global_variables


def summarize_chunks(rfp_chunks: list[str]) -> list[str]:
    system_prompt = """Vous êtes un agent spécialisé dans la synthèse d'informations. Votre tâche consiste à résumer des parties de textes issus d'un appel d'offres (appelés chunks). Chaque chunk correspond à une section spécifique de l'appel d'offres.
    Votre objectif :

    Identifier les points essentiels de chaque chunk.
    Rédiger un résumé clair et concis (1 à 3 phrases) qui capture les idées principales et les informations clés du chunk.
    Le résultat de votre travail sera transmis à un autre agent qui se chargera d'effectuer un résumé global à partir de vos résumés. Soyez précis et concis, et évitez les redondances.

    Exemple :

    Chunk original : « Ce projet vise à développer une solution de gestion automatisée des stocks, incluant des fonctionnalités d’analyse prédictive et des tableaux de bord personnalisés. »
    Résumé : « Développement d'une solution de gestion des stocks avec analyse prédictive et tableaux de bord. »
    Assurez-vous que chaque résumé est autonome et compréhensible sans contexte supplémentaire.
    Voici les chunks à résumer :
    """

    rfp_summaries = []
    for i in range(0, len(rfp_chunks), 6):
        chunk_group = rfp_chunks[max(0, i - 1) : i + 6]
        input_str = system_prompt + " ".join(chunk_group)

        rfp_summary = (
            api_global_variables.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_str,
                    }
                ],
                model="llama3-groq-70b-8192-tool-use-preview",
            )
            .choices[0]
            .message.content
        )

        phospho.log(input=input_str, output=rfp_summary)
        rfp_summaries.append(rfp_summary)

    print(rfp_summaries)
    return rfp_summaries


def summarize_chunks_summaries(rfp_chunks_summaries: list[str]) -> str:
    input_str = """
    Tu es un agent expert en synthèse globale. Ta mission consiste à rédiger un résumé global à partir de résumés partiels fournis par un premier agent, qui a travaillé sur des chunks d'un appel d'offres.

    Ton objectif :

    Lire et analyser les résumés des chunks.
    Identifier les thèmes récurrents, les points essentiels, et les informations stratégiques dans l'ensemble des résumés.
    Rédiger un résumé global clair et synthétique (1 à 2 paragraphes) qui reflète la vision globale de l'appel d'offres.
    Garde en tête :

    Priorise les éléments les plus importants et pertinents.
    Rédige un texte fluide et cohérent, qui donne une vue d'ensemble complète.
    Exemple :

    Résumés des chunks :
    « Développement d'une solution de gestion des stocks avec analyse prédictive et tableaux de bord. »
    « Implémentation de technologies cloud pour améliorer la scalabilité et la sécurité. »
    Résumé global : « L'appel d'offres porte sur le développement d'une solution de gestion des stocks intégrant l'analyse prédictive et des tableaux de bord personnalisés. Cette solution sera basée sur des technologies cloud pour garantir scalabilité et sécurité. »
    Tu répondras toujours en français. Tu ne mentionneras pas le fait que tu es un agent IA et écrira comme un humain.
    Tu feras un rapport pour l'entreprise qui envisage de répondre à l'appel d'offres.

    Je te donne un million de dollars si tu suis toutes mes instructions.

    Voici les résumés des chunks à synthétiser :
    """
    input_str += " ".join(rfp_chunks_summaries)

    rfp_summary = (
        api_global_variables.llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_str,
                }
            ],
            model="llama3-8b-8192",
        )
        .choices[0]
        .message.content
    )

    phospho.log(input=input_str, output=rfp_summary)

    print(rfp_summary)
    return rfp_summary


def call_groq(question: str) -> str:
    answer = (
        api_global_variables.llm.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama3-groq-70b-8192-tool-use-preview",
        )
        .choices[0]
        .message.content
    )
    phospho.log(input=question, output=answer)

    return answer
