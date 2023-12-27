from lambdaLearn.Evaluation.Cluster.Davies_Bouldin_Score import Davies_Bouldin_Score
from lambdaLearn.Evaluation.Cluster.Fowlkes_Mallows_Score import Fowlkes_Mallows_Score
from lambdaLearn.Evaluation.Cluster.Jaccard_Score import Jaccard_Score
from lambdaLearn.Evaluation.Cluster.Rand_Score import Rand_Score
from lambdaLearn.Evaluation.Cluster.Silhouette_Score import Silhouette_Score

k=3
tolerance=1e-7
max_iterations=300
evaluation={
    'Fowlkes_Mallows_Score':Fowlkes_Mallows_Score(),
    'Jaccard_Score':Jaccard_Score(average='macro'),
    'Rand_Score':Rand_Score(),
    'Davies_Bouldin_Score':Davies_Bouldin_Score(),
    'Silhouette_Score':Silhouette_Score()
}
verbose=False
file=None