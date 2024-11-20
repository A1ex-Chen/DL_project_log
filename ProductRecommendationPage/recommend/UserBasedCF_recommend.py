def recommend(self, users_score_item):
    recommendations = {}
    nearest = self.get_k_neighbor(users_score_item)
    userRatings = self.data[users_score_item]
    totalDistance = 0.0
    for i in range(self.k):
        totalDistance += nearest[i][1]
    if totalDistance == 0.0:
        totalDistance = 1.0
    for i in range(self.k):
        weight = nearest[i][1] / totalDistance
        name = nearest[i][0]
        neighborRatings = self.data[name]
        for cur_book in neighborRatings:
            if cur_book not in userRatings:
                if cur_book not in recommendations:
                    recommendations[cur_book] = neighborRatings[cur_book
                        ] * weight
                else:
                    recommendations[cur_book] = recommendations[cur_book
                        ] + neighborRatings[cur_book] * weight
    recommendations = list(recommendations.items())
    recommendations.sort(key=lambda cur_bookTuple: cur_bookTuple[1],
        reverse=True)
    return recommendations[:self.n], nearest
