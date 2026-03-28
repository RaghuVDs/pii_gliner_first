import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { message } from "antd";
import { useAuthStore } from "@/store/authStore";
import { useTenantStore } from "@/store/tenantStore";
import { authApi } from "@/api/auth";

export function useAuth() {
  const navigate = useNavigate();
  const {
    user,
    isAuthenticated,
    login: setAuth,
    logout: clearAuth,
  } = useAuthStore();

  const login = useCallback(
    async (email: string, password: string) => {
      try {
        const response = await authApi.login(email, password);
        // Build a minimal user object from the token response
        const tokenUser = {
          id: response.user_id || "",
          email,
          tenant_id: response.tenant_id || "",
        } as any;
        // Save token directly to localStorage for immediate availability
        localStorage.setItem("pii_access_token", response.access_token);
        localStorage.setItem("pii_refresh_token", response.refresh_token || "");
        setAuth(tokenUser, response.access_token, response.refresh_token || "");
        message.success("Logged in successfully");
        navigate("/dashboard");
      } catch (error: unknown) {
        const err = error as { response?: { data?: { detail?: string } } };
        message.error(err.response?.data?.detail || "Login failed");
        throw error;
      }
    },
    [navigate, setAuth]
  );

  const register = useCallback(
    async (
      email: string,
      password: string,
      fullName: string,
      orgName: string
    ) => {
      try {
        const response = await authApi.register(
          email,
          password,
          fullName,
          orgName
        );
        const tokenUser = {
          id: response.user_id || "",
          email,
          full_name: fullName,
          tenant_id: response.tenant_id || "",
        } as any;
        // Save token directly to localStorage for immediate availability
        localStorage.setItem("pii_access_token", response.access_token);
        localStorage.setItem("pii_refresh_token", response.refresh_token || "");
        setAuth(tokenUser, response.access_token, response.refresh_token || "");
        message.success("Registration successful");
        navigate("/dashboard");
      } catch (error: unknown) {
        const err = error as { response?: { data?: { detail?: string } } };
        message.error(err.response?.data?.detail || "Registration failed");
        throw error;
      }
    },
    [navigate, setAuth]
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // proceed with local logout even if API call fails
    } finally {
      localStorage.removeItem("pii_access_token");
      localStorage.removeItem("pii_refresh_token");
      clearAuth();
      useTenantStore.getState().clearTenant();
      navigate("/login");
    }
  }, [navigate, clearAuth]);

  return {
    user,
    isAuthenticated,
    login,
    register,
    logout,
  };
}
