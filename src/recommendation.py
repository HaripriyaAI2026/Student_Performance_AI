def get_recommendations(predicted_score, support_level):
    recommendations = []

    if predicted_score < 10:
        recommendations.append("Provide additional academic support and regular monitoring.")
        recommendations.append("Encourage extra study time and revision.")
        recommendations.append("Consider additional paid or school-supported classes.")
    elif predicted_score < 14:
        recommendations.append("Maintain regular study habits and monitor progress.")
        recommendations.append("Provide extra support in difficult subjects.")
    else:
        recommendations.append("Continue the current study routine.")
        recommendations.append("Encourage participation in extracurricular activities.")
        recommendations.append("Set higher academic goals.")

    if support_level == "High Support":
        recommendations.append("Regular guidance from teachers and family is recommended.")
    elif support_level == "Medium Support":
        recommendations.append("Periodic teacher and family monitoring is recommended.")

    return recommendations


def recommend(predicted_score, support_level):
    return get_recommendations(predicted_score, support_level)

