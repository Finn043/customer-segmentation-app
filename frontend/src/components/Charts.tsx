import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import type { CustomerPoint, DashboardData } from '../types'

const colors = ['#7357e8', '#f3a64a', '#35a989', '#ed6d79', '#4c9de0', '#ae71c8', '#e17d42', '#6e8498', '#d29b34', '#5dbaaf']
const number = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })

export function ClusterChart({ points, k }: { points: CustomerPoint[]; k: number }) {
  return <div className="h-[310px] w-full sm:h-[365px]"><ResponsiveContainer>
    <ScatterChart margin={{ top: 12, right: 18, bottom: 4, left: -18 }}><CartesianGrid strokeDasharray="3 5" vertical={false}/>
      <XAxis type="number" dataKey="pc1" name="PC1" tickLine={false} axisLine={false} label={{ value: 'Principal component 1', position: 'insideBottom', offset: -2, fill: '#a0a2aa', fontSize: 10 }}/>
      <YAxis type="number" dataKey="pc2" name="PC2" tickLine={false} axisLine={false} label={{ value: 'Principal component 2', angle: -90, position: 'insideLeft', fill: '#a0a2aa', fontSize: 10 }}/>
      <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => active && payload?.length ? <div className="rounded-xl border border-slate-100 bg-white p-3 text-xs shadow-xl"><b>Customer #{payload[0].payload.id}</b><div>Age {payload[0].payload.age} · Income ${payload[0].payload.income}k</div><div>Spending {payload[0].payload.spending} · Cluster {payload[0].payload.cluster + 1}</div></div> : null}/>
      {Array.from({ length: k }, (_, cluster) => <Scatter key={cluster} name={`Cluster ${cluster + 1}`} data={points.filter(p => p.cluster === cluster)} fill={colors[cluster % colors.length]} fillOpacity={0.8} r={5}/>)}
    </ScatterChart></ResponsiveContainer></div>
}

export function ElbowChart({ data }: { data: DashboardData['elbow'] }) {
  return <div className="h-[188px] w-full"><ResponsiveContainer><LineChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}><CartesianGrid vertical={false} strokeDasharray="3 5"/><XAxis dataKey="k" tickLine={false} axisLine={false} tickFormatter={v => `K${v}`}/><YAxis tickLine={false} axisLine={false} tickFormatter={number}/><Tooltip formatter={(value) => [Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 }), 'SSE']} labelFormatter={v => `K = ${v}`}/><Line type="monotone" dataKey="sse" stroke="#7357e8" strokeWidth={2.5} dot={{ r: 3, fill: '#7357e8', strokeWidth: 0 }} activeDot={{ r: 5 }}/></LineChart></ResponsiveContainer></div>
}
