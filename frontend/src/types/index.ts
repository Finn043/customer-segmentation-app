export interface CustomerPoint { id: number; pc1: number; pc2: number; cluster: number; age: number; income: number; spending: number }
export interface ClusterSummary { cluster: number; size: number; income: number; spending: number; age: number }
export interface DashboardData { points: CustomerPoint[]; clusters: ClusterSummary[]; elbow: { k: number; sse: number }[]; silhouette: number | null; variance_explained: number; k: number }
