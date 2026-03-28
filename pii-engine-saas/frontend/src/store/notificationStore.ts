import { create } from "zustand";
import { Notification } from "@/types/models";
import { get, patch, post } from "@/api/client";

interface NotificationState {
  notifications: Notification[];
  unreadCount: number;
  loading: boolean;

  fetchNotifications: () => Promise<void>;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
}

export const useNotificationStore = create<NotificationState>((set, getState) => ({
  notifications: [],
  unreadCount: 0,
  loading: false,

  fetchNotifications: async () => {
    set({ loading: true });
    try {
      const notifications = await get<Notification[]>("/notifications");
      const unreadCount = notifications.filter((n) => !n.is_read).length;
      set({ notifications, unreadCount, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  markRead: async (id: string) => {
    try {
      await patch(`/notifications/${id}`, { is_read: true });
      const notifications = getState().notifications.map((n) =>
        n.id === id ? { ...n, is_read: true } : n
      );
      const unreadCount = notifications.filter((n) => !n.is_read).length;
      set({ notifications, unreadCount });
    } catch {
      // silently fail
    }
  },

  markAllRead: async () => {
    try {
      await post("/notifications/mark-all-read");
      const notifications = getState().notifications.map((n) => ({
        ...n,
        is_read: true,
      }));
      set({ notifications, unreadCount: 0 });
    } catch {
      // silently fail
    }
  },
}));
