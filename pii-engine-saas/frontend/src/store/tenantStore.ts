import { create } from "zustand";
import { Tenant } from "@/types/models";
import { get, patch } from "@/api/client";

interface TenantState {
  tenant: Tenant | null;
  loading: boolean;

  fetchTenant: () => Promise<void>;
  updateTenant: (data: Partial<Tenant>) => Promise<void>;
  clearTenant: () => void;
}

export const useTenantStore = create<TenantState>((set) => ({
  tenant: null,
  loading: false,

  fetchTenant: async () => {
    set({ loading: true });
    try {
      const tenant = await get<Tenant>("/tenant");
      set({ tenant, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  updateTenant: async (data: Partial<Tenant>) => {
    set({ loading: true });
    try {
      const tenant = await patch<Tenant>("/tenant", data);
      set({ tenant, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  clearTenant: () => set({ tenant: null }),
}));
