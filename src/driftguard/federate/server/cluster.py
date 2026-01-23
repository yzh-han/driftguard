from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional, TypeVar
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from driftguard.federate.observation import Fp
from driftguard.federate.params import Params


@dataclass
class Group:
    """Client group with a prototype fingerprint and optional parameters.

    Attributes:
        clients: Client indices in the group.
        proto: Prototype fingerprint for the group.
        params: Group-level model parameters.
    """

    clients: List[int]

    def __post_init__(self):
        """Initialize optional attributes for type checking."""
        self.proto: Fp
        self.params: Params = []

        self._waitlist: List[int] = []

    def __eq__(self, other: object) -> bool:
        """Compare groups by client membership."""
        if not isinstance(other, Group):
            return False
        return self.clients == other.clients

    def __hash__(self) -> int:
        """Hash groups by client membership."""
        return hash(tuple(self.clients))


    @staticmethod
    def from_raw(
        clu_raw: List[int] | np.ndarray,
    ) -> List[Group]:
        """Convert raw cluster labels into Group instances.

        Args:
            clu_raw: Cluster labels for each client index.

        Returns:
            List of Group instances.
        """
        return [
            Group(np.where(clu_raw == gid)[0].tolist()) for gid in np.unique(clu_raw)
        ]  # [Group, ...]

    @property
    def size(self) -> int:
        """Return the number of clients in the group."""
        return len(self.clients)

    def from_old(self, old_group: Group) -> None:
        """Copy parameters from a previous group.

        Args:
            old_group: Previously aligned group.
        """
        self.params = old_group.params

    def proto_cid(self, D: np.ndarray) -> int:
        """Select the central client in the group based on distances.

        Args:
            D: Pairwise distance matrix for all clients.

        Returns:
            Client index representing the group's prototype.
        """
        if len(self.clients) == 1:
            return self.clients[0]
        D_group = D[np.ix_(self.clients, self.clients)]
        avg = D_group.mean(axis=1)
        return self.clients[int(np.argmin(avg))]  # -> cid

@dataclass
class ClusterArgs:
    """Arguments for clustering configuration."""
    thr: float = 0.5  # Distance threshold for clustering
    min_group_size: int = 3  # Minimum size for each group
    match_thr: float | None = None  # Distance threshold for aligning old and new groups
    w_size: int = 3  # Smoothing factor for weight calculation

    def __post_init__(self):
        self.match_thr = self.match_thr or 0.6 * self.thr

class GroupState:
    """Manage clustering state across rounds."""

    def __init__(
        self, num_clients: int, args: ClusterArgs = ClusterArgs()
    ):
        """Initialize clustering state.

        Args:
            thr: Distance threshold for clustering.
            min_group_size: Minimum size for each group.
            match_thr: Distance threshold for aligning old and new groups.
        """
        self._min_group_size = args.min_group_size
        self._match_thr = args.match_thr or 0.6 * args.thr  # 对齐阈值，默认0.6倍聚类阈值
        self._w_size = args.w_size
        self._num_clients = num_clients

        self._model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",  # 推荐 average/complete
            distance_threshold=args.thr,
        )

        self.groups: List[Group] = [
            Group(clients=[cid for cid in range(num_clients)])
        ]  # 初始单一组，包含所有客户端
    @property
    def all_clients(self) -> List[int]:
        """Return a list of all client indices."""
        return [cid for cid in range(self._num_clients)]
    
    def unique_groups(self, selection: List[int]) -> List[Group]:
        """Collect unique groups referenced by client selection."""
        groups: List[Group] = list(
            set(self.get_group(cid) for cid in selection)
        )
        return groups
    
    def get_group(self, cid: int, groups: List[Group] | None = None) -> Group:
        """Find the group containing a client."""
        groups = groups or self.groups

        for group in groups:
            if cid in group.clients:
                return group
        raise ValueError("Client ID not found in any group.")
        
    def update(self, fps: List[Fp]) -> None:
        """Cluster fingerprints and update group state.

        Args:
            fps: List of fingerprints for all clients.
        """
        # 0 计算距离矩阵
        D = Fp.pairwise_D(fps)  # 本轮30个
        # 1 聚类
        clu_raw = self._model.fit_predict(D)
        # 2 初始分组并设置原型
        groups = Group.from_raw(clu_raw)
        for g in groups:
            g.proto = fps[g.proto_cid(D)]
        # 3 对齐簇
        groups =self._align(groups)
        # 4 合并小类
        groups = self._merge(groups, D)
        # 5 更新原型
        for g in groups:
            g.proto = fps[g.proto_cid(D)]
        self.groups = groups

    def _merge(
        self,
        groups: List[Group],
        D: np.ndarray,
    ) -> List[Group]:
        """Merge small clusters into nearest larger clusters.

        Args:
            groups: Group list to merge.
            D: Pairwise distance matrix for all clients.

        Returns:
            Merged list of groups.
        """
        while True:
            # 找到最小的簇
            gidx_min, g_min = min(enumerate(groups), key=lambda i_g: i_g[1].size)

            if g_min.size >= self._min_group_size:
                break  # 全部满足最小簇大小，结束

            # 所有簇的原型
            proto_cids: List[int] = [g.proto_cid(D) for g in groups]  # [cid, ...]
            # 非原型 inf
            D_proto = np.full(D.shape, np.inf)
            D_proto[np.ix_(proto_cids, proto_cids)] = D[np.ix_(proto_cids, proto_cids)]
            D_proto[:, proto_cids[gidx_min]] = np.inf
            proto_nearest = np.argmin(D_proto[proto_cids[gidx_min], :])

            # 找到最近的簇
            g_nearest = self.get_group(int(proto_nearest), groups)
            gidx_nearest = groups.index(g_nearest)

            # 合并簇
            clients = groups.pop(gidx_min).clients
            groups[gidx_nearest].clients.extend(clients)

        return groups

    def _align(self, new_groups: List[Group]) -> List[Group]:
        """Align new clusters with existing groups by prototype distance.

        Args:
            groups: Newly computed groups for the current round.
        """
        if not self.groups:
            return new_groups

        old_groups = list(self.groups)
        new_groups = new_groups
        groups = []
        # 对齐已有簇

        # 计算簇间距离 [old, new]
        fgs: List[Fp] = [g.proto for g in old_groups + new_groups]
        D = Fp.pairwise_D(fgs)[
            np.ix_(range(len(old_groups)), range(len(old_groups), len(fgs)))
        ]  # [old, new]

        while True:
            if D.size == 0 or D.min() > self._match_thr:
                break

            # 找最近的簇对
            old_idx, new_idx = np.unravel_index(np.argmin(D), D.shape)
            new_groups[new_idx].from_old(old_groups[old_idx]) # 传递参数
            groups.append(new_groups[new_idx])

            # 删除已对齐的簇
            D = np.delete(D, old_idx, axis=0)
            D = np.delete(D, new_idx, axis=1)
            old_groups.pop(old_idx)
            new_groups.pop(new_idx)

        # 剩余的新簇直接加入
        groups.extend(new_groups)
        return groups
