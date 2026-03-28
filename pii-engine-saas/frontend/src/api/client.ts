import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  InternalAxiosRequestConfig,
} from "axios";
import { useAuthStore } from "@/store/authStore";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// Request interceptor: attach JWT token
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Read token - check localStorage first (synchronous, reliable even before
    // Zustand persist hydration completes), then fall back to Zustand store.
    const token =
      localStorage.getItem("pii_access_token") ||
      useAuthStore.getState().accessToken;

    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: handle 401 and token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      // Check both localStorage and Zustand for tokens to determine if the
      // user was actually authenticated before this request.
      const storedToken = localStorage.getItem("pii_access_token");
      const storedRefresh = localStorage.getItem("pii_refresh_token");
      const { refreshToken, accessToken } = useAuthStore.getState();

      const hadToken = storedToken || accessToken;
      const activeRefresh = storedRefresh || refreshToken;

      // If no tokens exist, user isn't logged in - just reject silently
      if (!hadToken && !activeRefresh) {
        return Promise.reject(error);
      }

      if (activeRefresh) {
        try {
          const response = await axios.post(`${BASE_URL}/auth/refresh`, {
            refresh_token: activeRefresh,
          });

          const { access_token, refresh_token: newRefreshToken } =
            response.data;

          // Persist to both localStorage and Zustand
          localStorage.setItem("pii_access_token", access_token);
          localStorage.setItem("pii_refresh_token", newRefreshToken || "");
          useAuthStore.getState().setTokens(access_token, newRefreshToken);

          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return apiClient(originalRequest);
        } catch {
          // Refresh failed - session truly expired, only logout if user was authenticated
          if (hadToken) {
            localStorage.removeItem("pii_access_token");
            localStorage.removeItem("pii_refresh_token");
            useAuthStore.getState().logout();
            window.location.href = "/login";
          }
          return Promise.reject(error);
        }
      }
    }

    return Promise.reject(error);
  }
);

// Generic request methods
export async function get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.get<T>(url, config);
  return response.data;
}

export async function post<T>(
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.post<T>(url, data, config);
  return response.data;
}

export async function patch<T>(
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.patch<T>(url, data, config);
  return response.data;
}

export async function del<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.delete<T>(url, config);
  return response.data;
}

export default apiClient;
