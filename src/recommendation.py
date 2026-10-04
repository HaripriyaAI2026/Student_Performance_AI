def generate_recommendations(student_data):
    recommendations = []

    if "absences" in student_data:
        if student_data["absences"] >= 10:
            recommendations.append(
                "Improve attendance and maintain a regular study schedule."
            )
        elif student_data["absences"] >= 5:
            recommendations.append(
                "Try to reduce absences and attend classes more consistently."
            )

    if "failures" in student_data:
        if student_data["failures"] >= 2:
            recommendations.append(
                "Review previously difficult subjects and practice basic concepts."
            )
        elif student_data["failures"] == 1:
            recommendations.append(
                "Spend additional time revising topics where you previously struggled."
            )

    if "studytime" in student_data:
        if student_data["studytime"] <= 2:
            recommendations.append(
                "Increase weekly study time using short, consistent study sessions."
            )
        else:
            recommendations.append(
                "Continue the current study routine and focus on weak topics."
            )

    if "goout" in student_data and student_data["goout"] >= 4:
        recommendations.append(
            "Balance social activities with a structured academic schedule."
        )

    if "Dalc" in student_data and student_data["Dalc"] >= 3:
        recommendations.append(
            "Maintain healthy routines and minimize factors that may affect study consistency."
        )

    if "Walc" in student_data and student_data["Walc"] >= 3:
        recommendations.append(
            "Maintain healthy routines and protect regular study time."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain your current learning routine and regularly review your progress."
        )

    return recommendations