from pydantic_ai import Agent

from common.api_global_variables import api_global_variables


import requests
from bs4 import BeautifulSoup


@Agent.tool
def web_search(requete: str) -> str:
    """
    Effectue une recherche internet et retourne les descriptions concaténées des 3 premiers résultats.

    Args:
        requete (str): Le terme de recherche

    Returns:
        str: Descriptions concaténées des résultats de recherche
    """
    # URL de base pour DuckDuckGo
    url = f"https://duckduckgo.com/html/?q={requete.replace(' ', '+')}"

    # En-têtes pour simuler un navigateur
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Effectuer la requête
        reponse = requests.get(url, headers=headers)
        reponse.raise_for_status()

        # Parser le HTML
        soup = BeautifulSoup(reponse.text, "html.parser")

        # Trouver les résultats
        resultats = soup.find_all("div", class_="result__body")

        # Extraire et concaténer les descriptions
        descriptions = []
        for resultat in resultats[:3]:
            description = resultat.find("a", class_="result__snippet")
            if description:
                descriptions.append(description.text.strip())

        # Concaténer les descriptions
        return " ".join(descriptions)

    except requests.RequestException as e:
        print(f"Erreur de recherche : {e}")
        return ""


@Agent.tool
async def natural_language_search(query: str, rfp_id: str) -> str:
    """
    Cherche les parties les plus pertinentes de l'appel d'offre qui correspondent à la requête
    """

    query_embedding = api_global_variables.embedder.embed(query)

    search_results = api_global_variables.qdrant_client.search(
        collection_name=rfp_id, query_vector=query_embedding, limit=4
    )

    results_text = (
        "Voici les parties de l'appel d'offre qui correspondent à votre recherche:\n"
    )

    for result in search_results:
        points = result.payload
        previous_chunk = str(points.get("previous_chunk", ""))
        chunk = str(points.get("chunk", ""))
        next_chunk = str(points.get("next_chunk", ""))
        results_text += f"{previous_chunk}\n{chunk}\n{next_chunk}\n\n"

    return results_text.strip()
